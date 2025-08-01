from .get_full_grad_list import get_full_grad_list
from .state import TrainingState, TrainingResults
import math

def train(state: TrainingState, results: TrainingResults, trainloader, trainloader_full, total_steps, max_bs):
    state.model.train()
    train_loss = 0
    correct = 0
    total = 0

    bs = state.bs_scheduler.get_batch_size()
    lr = state.lr_scheduler.get_last_lr()[0]
    eps = state.eps

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

        if state.lr_step_type == 'steps':
            state.lr_scheduler.step()
            lr = state.lr_scheduler.get_last_lr()[0]

        if total_steps is not None and state.steps.total >= total_steps:
            break

    grad_norm = get_full_grad_list(state.model, trainloader_full, state.optimizer, state.device)
    norm_result = [state.epoch + 1, state.steps.total, grad_norm]
    results.norm.append(norm_result)

    train_accuracy = 100. * correct / total
    train_result = [state.epoch + 1, state.steps.total, train_loss / (batch_idx + 1), train_accuracy, lr]
    results.train.append(train_result)

    if eps is not None and grad_norm <= eps and bs < max_bs:
        state.steps.reset_current_bs()
        if state.bs_step_type == 'eps':
            state.bs_scheduler.step()
            bs = state.bs_scheduler.get_batch_size()
        if state.lr_step_type == 'eps':
            state.lr_scheduler.step()
            lr = state.lr_scheduler.get_last_lr()[0]
        state.eps = eps / math.sqrt(1.4)
        eps = state.eps
        print(f'eps: {eps}')
        print(f'  -> grad_norm <= eps -> increase batch size & lr: batch_size={bs}, learning_rate={lr:.5f}')

    if state.lr_step_type == 'epoch':
        state.lr_scheduler.step()
