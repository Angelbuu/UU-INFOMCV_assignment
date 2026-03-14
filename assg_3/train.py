import os
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import confusion_matrix

from models import init_model, LeNet, LeNetVariant1, LeNetVariant2
from models import CIFAR100Model, CIFAR100LeNet, CIFAR100Variant1
from import_data import load_data


def minibatch_forward_pass(model, minibatch, loss_function):
    images, labels = minibatch
    outputs = model(images)

    loss = loss_function(outputs, labels)
    predictions = outputs.argmax(dim=1)

    batch_loss = loss.item()
    batch_size = labels.size(0)
    batch_correct = (predictions == labels).sum().item()

    return loss, batch_loss, batch_size, batch_correct


def validate(model: nn.Module, val_data):
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    val_loss = 0
    num_labels = 0
    correct_predictions = 0

    with torch.no_grad():
        for minibatch in val_data:
            _, batch_loss, batch_size, batch_correct = minibatch_forward_pass(model, minibatch, loss_function)

            val_loss += batch_loss
            num_labels += batch_size
            correct_predictions += batch_correct

    accuracy = 100 * correct_predictions / num_labels
    return val_loss, accuracy


def train(model, train_data, val_data, max_epochs=50, optim=Adam, lr=0.001, patience=5):
    print('\nTraining', type(model).__name__)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        correct, total = 0, 0

        for batch in train_data:
            opt.zero_grad()
            loss, bl, bs, bc = minibatch_forward_pass(model, batch, loss_fn)
            loss.backward()
            opt.step()
            epoch_loss += bl
            correct += bc
            total += bs

        train_acc = 100 * correct / total
        val_loss, val_acc = validate(model, val_data)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}  train loss: {epoch_loss:.2f} acc: {train_acc:.1f}%  val loss: {val_loss:.2f} acc: {val_acc:.1f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'early stop at epoch {epoch+1}')
                break

    return model, train_losses, train_accs, val_losses, val_accs


def evaluate(model, test_data):
    """Returns accuracy and (all_preds, all_labels) for confusion matrix."""
    model.eval()
    preds, labels_list = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, lbls in test_data:
            out = model(imgs)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            labels_list.extend(lbls.cpu().numpy())
            correct += (out.argmax(dim=1) == lbls).sum().item()
            total += lbls.size(0)

    return 100 * correct / total, preds, labels_list


def copy_fc_for_pretrained(old_fc, num_outs=10):
    """New fc with 10 outputs, weights copied from first 10 of the 20-output layer."""
    new_fc = nn.Linear(84, num_outs)
    with torch.no_grad():
        new_fc.weight.copy_(old_fc.weight[:num_outs])
        new_fc.bias.copy_(old_fc.bias[:num_outs])
    return new_fc


def main():
    os.makedirs('checkpoints', exist_ok=True)

    cif_ten_train, cif_ten_val, test_data = load_data(val_set_ratio=0.15)
    cif_hundred_train, cif_hundred_val, _ = load_data(dataset='CIF_HUNDRED', val_set_ratio=0.15)

    results = []

    # train three CIFAR-10 models
    lenet, tl, ta, vl, va = train(init_model(LeNet), cif_ten_train, cif_ten_val)
    torch.save(lenet.state_dict(), 'checkpoints/CIFAR10_lenet.pt')
    results.append(('CIFAR10_lenet', tl[-1], ta[-1], vl[-1], va[-1]))

    lenet_v1, tl, ta, vl, va = train(init_model(LeNetVariant1), cif_ten_train, cif_ten_val)
    torch.save(lenet_v1.state_dict(), 'checkpoints/CIFAR10_model1.pt')
    results.append(('CIFAR10_model1', tl[-1], ta[-1], vl[-1], va[-1]))

    lenet_v2, tl, ta, vl, va = train(init_model(LeNetVariant2), cif_ten_train, cif_ten_val)
    torch.save(lenet_v2.state_dict(), 'checkpoints/CIFAR10_model2.pt')
    results.append(('CIFAR10_model2', tl[-1], ta[-1], vl[-1], va[-1]))

    # pick best by val accuracy
    best_idx = max(range(3), key=lambda i: results[i][4])
    best_name = results[best_idx][0]
    print(f'\nbest CIFAR-10 model: {best_name} (val acc {results[best_idx][4]:.1f}%)')

    cifar100_cls = CIFAR100LeNet if best_name == 'CIFAR10_lenet' else (CIFAR100Variant1 if best_name == 'CIFAR10_model1' else CIFAR100Model)
    cif_100 = init_model(cifar100_cls)
    train(cif_100, cif_hundred_train, cif_hundred_val)
    torch.save(cif_100.state_dict(), 'checkpoints/CIFAR100_model.pt')

    # CIFAR10_pretrained: replace last layer, copy first 10 weights, fine-tune with lr/2
    base_cls = LeNet if best_name == 'CIFAR10_lenet' else (LeNetVariant1 if best_name == 'CIFAR10_model1' else LeNetVariant2)
    cif_10_pretrained = init_model(base_cls)
    sd = {k: v for k, v in cif_100.state_dict().items() if k != 'fc_2.weight' and k != 'fc_2.bias'}
    cif_10_pretrained.load_state_dict(sd, strict=False)
    cif_10_pretrained.fc_2 = copy_fc_for_pretrained(cif_100.fc_2, 10)
    train(cif_10_pretrained, cif_ten_train, cif_ten_val, lr=0.0005)
    torch.save(cif_10_pretrained.state_dict(), 'checkpoints/CIFAR10_pretrained.pt')

    # final comparison on test set
    best_model = lenet_v2 if best_name == 'CIFAR10_model2' else (lenet_v1 if best_name == 'CIFAR10_model1' else lenet)
    acc_best, preds_best, lbls = evaluate(best_model, test_data)
    acc_pre, preds_pre, _ = evaluate(cif_10_pretrained, test_data)

    print('\n--- Performance table (last epoch) ---')
    for name, tl, ta, vl, va in results:
        print(f'{name}: train loss {tl:.2f} acc {ta:.1f}%  val loss {vl:.2f} acc {va:.1f}%')

    print('\n--- Test set comparison ---')
    print(f'{best_name}: {acc_best:.1f}%')
    print(f'CIFAR10_pretrained: {acc_pre:.1f}%')

    print('\nConfusion matrix (best):')
    print(confusion_matrix(lbls, preds_best))
    print('Confusion matrix (pretrained):')
    print(confusion_matrix(lbls, preds_pre))


if __name__ == '__main__':
    main()
