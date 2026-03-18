from models import init_model, LeNet
from train import train
from import_data import load_data


def main():
    cif_ten_train, cif_ten_val, test_data = load_data(val_set_ratio=0.15)
    lenet, tl, ta, vl, va, _ = train(init_model(LeNet), cif_ten_train, cif_ten_val, use_feedback=True)


if __name__ == '__main__':
    main()
