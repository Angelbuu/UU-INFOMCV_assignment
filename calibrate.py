import numpy as np
import cv2 as cv

# 1. LOAD THE DATA
# Load your 20 automatic detections (make sure you saved objpoints/imgpoints here)
try:
    auto_data = np.load('B.npz') 
    obj_auto = list(auto_data['objpoints'])
    img_auto = list(auto_data['imgpoints'])
except FileNotFoundError:
    print("Error: B_points.npz not found. Ensure you saved raw points in preprocessing.")

# Load your partner's 5 manual detections
try:
    manual_data = np.load('F.npz')
    obj_manual = list(manual_data['objpoints'])
    img_manual = list(manual_data['imgpoints'])
except FileNotFoundError:
    print("Error: F.npz not found.")

# Standard image resolution (Width, Height)
img_size = (4284, 5712)


def reject_bad_image(op, ip, base_rms, mtx, dist, rvecs, tvecs, epsilon=0.1):
    if len(op) == 1:
        return base_rms, mtx, dist, rvecs, tvecs
    lowest_rms = np.inf
    bad_image_idx = -1
    for i in range(len(op)):
        rms, new_mtx, new_dist, new_rvecs, new_tvecs = cv.calibrateCamera(
            op[:i] + op[i+1:], ip[:i] + ip[i+1:], img_size, None, None, flags=0
        )
        if rms < base_rms - epsilon and rms < lowest_rms:
            lowest_rms = rms
            mtx, dist, rvecs, tvecs = new_mtx, new_dist, new_rvecs, new_tvecs
            bad_image_idx = i
    if lowest_rms > base_rms:
        return base_rms, mtx, dist, rvecs, tvecs
    else:
        print('Rejected bad image, number of remaining images:', len(op) - 1)
        print(f'Previous rms {base_rms} vs new rms {lowest_rms}')
        return reject_bad_image(op[:bad_image_idx] + op[bad_image_idx+1:],
                                ip[:bad_image_idx] + ip[bad_image_idx+1:],
                                lowest_rms, mtx, dist, rvecs, tvecs, epsilon)


# 2. CALIBRATION FUNCTION
def run_calibration_experiment(op, ip, run_name, reject_bad_images=True): # Default to True for points
    """
    Performs calibration and implements Choice Task 2: 
    Iterative rejection of high-error images.
    """
    # First pass: Initial calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        op, ip, img_size, None, None, flags=0
    )

    # Choice Task 2: Iterative rejection
    if reject_bad_images:
        # We use a threshold of 1.0 pixels to filter out noisy manual clicks
        ret, mtx, dist, rvecs, tvecs = reject_bad_image(op, ip, ret, mtx, dist, rvecs, tvecs)
    
    print(f"\n--- {run_name} Results ---")
    print(f"Final Reprojection Error (RMS): {ret:.5f} pixels")
    print("Camera Matrix (K):")
    print(np.array2string(mtx, precision=4, suppress_small=True))
    
    return mtx, dist

# 3. EXECUTE THE THREE RUNS

# 3. EXECUTE THE THREE RUNS
# Now all runs will use Choice Task 2 logic automatically 

# --- RUN 1: Full Dataset (25 imgs) ---
run1_obj = obj_auto + obj_manual
run1_img = img_auto + img_manual
mtx1, dist1 = run_calibration_experiment(run1_obj, run1_img, "Run 1", reject_bad_images=False)

# --- RUN 2: Balanced subset (10 imgs) ---
run2_obj = obj_auto[:5] + obj_manual
run2_img = img_auto[:5] + img_manual
mtx2, dist2 = run_calibration_experiment(run2_obj, run2_img, "Run 2", reject_bad_images=False)

# --- RUN 3: Minimum subset (5 imgs) ---
run3_obj = obj_auto[:5]
run3_img = img_auto[:5]
mtx3, dist3 = run_calibration_experiment(run3_obj, run3_img, "Run 3", reject_bad_images=False)

# Save the final "clean" matrices for the Online Phase
np.savez('final_calibration_results.npz', mtx=np.array([mtx1, mtx2, mtx3]), dist=np.array([dist1, dist2, dist3]))
print("\n✅ All 3 runs completed and saved with Choice Task 2 rejection.")
