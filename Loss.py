import torch


def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))


def dice_loss(y_real, y_pred):
    return 1 - (torch.mean(2 * torch.mul(y_real, y_pred) + 1) / (torch.mean(torch.add(y_real, y_pred)) + 1))

def cross_entropy_loss(y_real, y_pred):
        y_pred = torch.sigmoid(y_pred)
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(y_pred, y_real)
        
        return loss

def cross_entropy_weighted_loss(y_real, y_pred, pos_weight=1.0):
    y_pred = torch.sigmoid(y_pred)
    loss_fn = torch.nn.BCELoss(weight=(y_real * (pos_weight - 1) + 1))
    loss = loss_fn(y_pred, y_real)
    
    return loss

# Link explaining focal loss: https://medium.com/elucidate-ai/an-introduction-to-focal-loss-b49d18cb3ded
def focal_loss(y_real, y_pred, gamma_2=2.0):
    prob = torch.sigmoid(y_pred)
    pt = torch.where(y_real == 1, prob, 1 - prob)
    focal_loss = - (1 - pt) ** gamma_2 * torch.log(pt + 1e-8)
    
    return torch.mean(focal_loss)

import torch

def point_loss(y_pred, y_true, pos_points, neg_points):
    """
    Compute the point loss.

    :param y_pred: Tensor of predicted scores (N, C)
    :param y_true: Tensor of ground truth labels (N,) 
    :param pos_points: Tensor of positive point indices
    :param neg_points: Tensor of negative point indices
    :return: Computed point loss
    """
    # Gather ground truth labels at point locations
    pos_labels = y_true[pos_points]
    neg_labels = y_true[neg_points]

    # Gather predictions at point locations
    pos_predictions = y_pred[pos_points, pos_labels]  # Predictions for positive points
    neg_predictions = y_pred[neg_points, neg_labels]  # Predictions for negative points

    # Concatenate predictions
    predictions = torch.cat([pos_predictions, neg_predictions])

    # Calculate loss (using appropriate loss function like cross-entropy)
    loss = -torch.mean(torch.log(predictions + 1e-8))  # Adding epsilon for numerical stability

    return loss

# Test