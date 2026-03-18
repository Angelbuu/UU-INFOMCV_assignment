import os
import torch
import torch.nn as nn
from torch.optim import Adam

from models import init_model, LeNet, LeNetVariant1, LeNetVariant2
from models import CIFAR100Variant2, CIFAR100LeNet, CIFAR100Variant1
from import_data import load_data
from graphs import plot_metrics, plot_confusion_matrix


def minibatch_forward_pass(model, minibatch, loss_function, use_feedback=False):
    """One forward pass; returns loss, batch loss, size, correct count. use_feedback adds auxiliary losses."""
    images, labels = minibatch
    if use_feedback:
        outputs, fb_outputs_1, fb_outputs_2 = model(images, use_feedback=use_feedback)
        loss_out = loss_function(outputs, labels)
        loss_fb1 = loss_function(fb_outputs_1, labels)
        loss_fb2 = loss_function(fb_outputs_2, labels)
        loss = loss_out + 0.5 * loss_fb1 + 0.5 * loss_fb2
    else:
        outputs = model(images)
        loss = loss_function(outputs, labels)

    predictions = outputs.argmax(dim=1)

    batch_loss = loss.item()
    batch_size = labels.size(0)
    batch_correct = (predictions == labels).sum().item()

    return loss, batch_loss, batch_size, batch_correct


def validate(model, val_data):
    """Evaluates model on validation set; returns total loss and accuracy %."""
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


def train(model, train_data, val_data, max_epochs=50, optim=Adam, lr=0.001, patience=5, use_lr_schedule=False,
          use_feedback=False):
    """Trains model with early stopping; optional LR schedule (x0.5 every 5 epochs) and auxiliary feedback losses."""
    print('\nTraining', type(model).__name__, '(lr schedule)' if use_lr_schedule else '')

    loss_fn = nn.CrossEntropyLoss()
    opt = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5) if use_lr_schedule else None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lr_history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        lr_history.append(opt.param_groups[0]['lr'])
        model.train()
        epoch_loss = 0
        correct, total = 0, 0

        for batch in train_data:
            opt.zero_grad()
            loss, bl, bs, bc = minibatch_forward_pass(model, batch, loss_fn, use_feedback)
            loss.backward()
            opt.step()
            epoch_loss += bl
            correct += bc
            total += bs

        if scheduler is not None:
            scheduler.step()

        train_acc = 100 * correct / total
        val_loss, val_acc = validate(model, val_data)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}  lr: {opt.param_groups[0]["lr"]:.6f}  train loss: {epoch_loss:.2f} acc: {train_acc:.1f}%  val loss: {val_loss:.2f} acc: {val_acc:.1f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'early stop at epoch {epoch+1}')
                break

    return model, train_losses, train_accs, val_losses, val_accs, lr_history


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


def train_cifar_10_models(cif_ten_train, cif_ten_val):
    """Trains LeNet, Variant1, Variant2; saves checkpoints and plots; returns results and models."""
    results = []

    lenet, tl, ta, vl, va, _ = train(init_model(LeNet), cif_ten_train, cif_ten_val)
    torch.save(lenet.state_dict(), 'checkpoints/CIFAR10_lenet.pt')
    results.append(('CIFAR10_lenet', tl[-1], ta[-1], vl[-1], va[-1]))
    plot_metrics(tl, vl, ylabel='Loss', title='LeNet Loss over Epochs')
    plot_metrics(ta, va, ylabel='Accuracy', title='LeNet Accuracy over Epochs')

    lenet_v1, tl, ta, vl, va, _ = train(init_model(LeNetVariant1), cif_ten_train, cif_ten_val)
    torch.save(lenet_v1.state_dict(), 'checkpoints/CIFAR10_model1.pt')
    results.append(('CIFAR10_model1', tl[-1], ta[-1], vl[-1], va[-1]))
    plot_metrics(tl, vl, ylabel='Loss', title='LeNet Variant 1 Loss over Epochs')
    plot_metrics(ta, va, ylabel='Accuracy', title='LeNet Variant 1 Accuracy over Epochs')

    lenet_v2, tl, ta, vl, va, _ = train(init_model(LeNetVariant2), cif_ten_train, cif_ten_val)
    torch.save(lenet_v2.state_dict(), 'checkpoints/CIFAR10_model2.pt')
    results.append(('CIFAR10_model2', tl[-1], ta[-1], vl[-1], va[-1]))
    plot_metrics(tl, vl, ylabel='Loss', title='LeNet Variant 2 Loss over Epochs')
    plot_metrics(ta, va, ylabel='Accuracy', title='LeNet Variant 2 Accuracy over Epochs')

    return results, lenet, lenet_v1, lenet_v2


def train_cifar_100_model(best_name):
    """Trains CIFAR-100 model using best CIFAR-10 architecture (20 coarse classes)."""
    cif_hundred_train, cif_hundred_val, _ = load_data(dataset='CIF_HUNDRED', val_set_ratio=0.15)

    if best_name == 'CIFAR10_lenet':
        cifar100_cls = CIFAR100LeNet
    elif best_name == 'CIFAR10_model1':
        cifar100_cls = CIFAR100Variant1
    else:
        cifar100_cls = CIFAR100Variant2

    cif_100 = init_model(cifar100_cls)
    train(cif_100, cif_hundred_train, cif_hundred_val)
    torch.save(cif_100.state_dict(), 'checkpoints/CIFAR100_model.pt')

    return cif_100


def finetune(best_name, cif_100, cif_ten_train, cif_ten_val):
    """Loads CIFAR-100 weights, replaces last layer for 10 outputs, fine-tunes on CIFAR-10 with lr/2."""
    if best_name == 'CIFAR10_lenet':
        base_cls = LeNet
    elif best_name == 'CIFAR10_model1':
        base_cls = LeNetVariant1
    else:
        base_cls = LeNetVariant2

    cif_10_pretrained = init_model(base_cls)
    sd = {k: v for k, v in cif_100.state_dict().items() if k != 'fc_2.weight' and k != 'fc_2.bias'}
    cif_10_pretrained.load_state_dict(sd, strict=False)

    train(cif_10_pretrained, cif_ten_train, cif_ten_val, lr=0.0005)
    torch.save(cif_10_pretrained.state_dict(), 'checkpoints/CIFAR10_pretrained.pt')

    return cif_10_pretrained


def test_set_results(results, best_model, best_name, cif_10_pretrained, test_data):
    """Prints performance table and confusion matrices; saves confusion matrix plots."""
    acc_best, preds_best, lbls = evaluate(best_model, test_data)
    acc_pre, preds_pre, _ = evaluate(cif_10_pretrained, test_data)

    print('\nPerformance table (last epoch)')
    for name, tl, ta, vl, va in results:
        print(f'{name}: train loss {tl:.2f} acc {ta:.1f}%  val loss {vl:.2f} acc {va:.1f}%')

    print('\nTest set comparison')
    print(f'{best_name}: {acc_best:.1f}%')
    print(f'CIFAR10_pretrained: {acc_pre:.1f}%')

    plot_confusion_matrix(lbls, preds_best, title=f"{best_name} Confusion Matrix")
    plot_confusion_matrix(lbls, preds_pre, title="CIFAR10 Pretrained Confusion Matrix")


def main():
    """Full pipeline: train CIFAR-10 models, pick best, train CIFAR-100, fine-tune, compare on test set."""
    os.makedirs('checkpoints', exist_ok=True)

    cif_ten_train, cif_ten_val, test_data = load_data(val_set_ratio=0.15)

    results, lenet, lenet_v1, lenet_v2 = train_cifar_10_models(cif_ten_train, cif_ten_val)

    best_idx = max(range(3), key=lambda i: results[i][4])
    best_name = results[best_idx][0]
    print(f'\nbest CIFAR-10 model: {best_name} (val acc {results[best_idx][4]:.1f}%)')

    cif_100 = train_cifar_100_model(best_name)
    cif_10_pretrained = finetune(best_name, cif_100, cif_ten_train, cif_ten_val)

    if best_name == 'CIFAR10_lenet':
        best_model = lenet
    elif best_name == 'CIFAR10_model1':
        best_model = lenet_v1
    else:
        best_model = lenet_v2

    test_set_results(results, best_model, best_name, cif_10_pretrained, test_data)


if __name__ == '__main__':
    main()
