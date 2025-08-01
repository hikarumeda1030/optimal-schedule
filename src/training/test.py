import torch
from .state import TrainingState, TrainingResults


def test(state: TrainingState, results: TrainingResults, testloader):
    state.model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(state.device), labels.to(state.device)
            outputs = state.model(images)
            loss = state.criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    test_accuracy = 100. * correct / total
    test_result = [state.epoch + 1, state.steps.total, test_loss / (batch_idx + 1), test_accuracy]
    results.test.append(test_result)
