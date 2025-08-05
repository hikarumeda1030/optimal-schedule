from .get_full_grad_list import get_full_grad_list
from .state import TrainingState, TrainingResults


def train(state: TrainingState, results: TrainingResults, trainloader, trainloader_full):
    state.model.train()
    train_loss = 0
    correct = 0
    total = 0

    bs = state.bs_scheduler.get_batch_size()
    lr = state.lr_scheduler.get_last_lr()[0]

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(state.device), labels.to(state.device)

        state.optimizer.zero_grad()
        outputs = state.model(images)
        loss = state.criterion(outputs, labels)
        loss.backward()

        state.steps.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        results.lr_bs.append([state.epoch + 1, state.steps.total, lr, bs])

        state.optimizer.step()

    grad_norm = get_full_grad_list(state.model, trainloader_full, state.optimizer, state.device)
    norm_result = [state.epoch + 1, state.steps.total, grad_norm]
    results.norm.append(norm_result)

    train_accuracy = 100. * correct / total
    train_result = [state.epoch + 1, state.steps.total, train_loss / (batch_idx + 1), train_accuracy, lr]
    results.train.append(train_result)
