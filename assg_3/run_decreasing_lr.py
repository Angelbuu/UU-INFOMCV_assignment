import matplotlib.pyplot as plt

from import_data import load_data
from models import init_model, LeNet
from train import train, evaluate


def main():
    """CHOICE 1: LR schedule (halve every 5 epochs), plot, compare to baseline."""
    cif_ten_train, cif_ten_val, test_data = load_data(val_set_ratio=0.15)

    # baseline: no schedule
    print('=== Baseline (no LR schedule) ===')
    model_baseline = init_model(LeNet)
    _, tl_b, ta_b, vl_b, va_b, lr_b = train(model_baseline, cif_ten_train, cif_ten_val, use_lr_schedule=False)

    # with schedule: halve every 5 epochs
    print('\n=== With LR schedule (x0.5 every 5 epochs) ===')
    model_sched = init_model(LeNet)
    _, tl_s, ta_s, vl_s, va_s, lr_s = train(model_sched, cif_ten_train, cif_ten_val, use_lr_schedule=True)

    # plot learning rate over time
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(lr_b) + 1), lr_b, 'b-', label='baseline (constant)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.title('Baseline: constant LR')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(lr_s) + 1), lr_s, 'r-', label='schedule (x0.5 every 5)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.title('LR schedule')
    plt.legend()

    plt.tight_layout()
    plt.savefig('lr_schedule.png', dpi=150)
    print('\nSaved plot to lr_schedule.png')
    plt.close()

    # compare
    acc_b = evaluate(model_baseline, test_data)[0]
    acc_s = evaluate(model_sched, test_data)[0]
    print('\n--- Choice 1 comparison ---')
    print(f'Baseline (no schedule):  val acc {va_b[-1]:.1f}%  test acc {acc_b:.1f}%')
    print(f'With LR schedule:        val acc {va_s[-1]:.1f}%  test acc {acc_s:.1f}%')


if __name__ == '__main__':
    main()
