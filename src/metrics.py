import torch

### The following is the additional loss metrics
CLASS_NAMES = ["background", "chamber", "pores"]
EPSILON = 1e-12

# -----------------------------------------------------------------------------
# Core: Classwise Metrics (Sums, not percentages)
# These are the source of truth for all other metrics.
# -----------------------------------------------------------------------------

def classwise_true_positives(y_pred, y_true, weight=None):
    """
    Computes sum of the true positives along the batch/spatial axes.
    Returns: Raw true positive counts for each class (C,).
    """
    y_pred = y_pred.float()
    y_true = y_true.float()

    if weight is not None:
        weight = weight.float()
        pred_match = y_pred * weight
        true_match = y_true * weight
    else:
        pred_match = y_pred
        true_match = y_true

    # Sum over Batch(0), Height(2), Width(3) -> Result shape (Channel,)
    tp = torch.sum(pred_match * true_match, dim=[0, 2, 3]) 
    return tp

def classwise_false_positives(y_pred, y_true, weight=None):
    """Computes sum of the false positives (C,)."""
    return classwise_true_positives(y_pred, 1 - y_true, weight)

def classwise_false_negatives(y_pred, y_true, weight=None):
    """Computes sum of the false negatives (C,)."""
    return classwise_true_positives(1 - y_pred, y_true, weight)

def classwise_true_negatives(y_pred, y_true, weight=None):
    """Computes sum of the true negatives (C,)."""
    return classwise_true_positives(1 - y_pred, 1 - y_true, weight)


def classwise_iou(y_pred, y_true, weight=None):
    """
    Computes classwise Intersection over Union (IoU).
    Returns: Tensor with class-wise IoU scores (C,).
    """
    tp = classwise_true_positives(y_pred, y_true, weight)
    fp = classwise_false_positives(y_pred, y_true, weight)
    fn = classwise_false_negatives(y_pred, y_true, weight)

    ious = tp / (tp + fp + fn + EPSILON)
    return ious

def classwise_dice(y_pred, y_true, weight=None):
    """
    Computes classwise Dice coefficient.
    Returns: Tensor with class-wise Dice scores (C,).
    """
    tp = classwise_true_positives(y_pred, y_true, weight)
    fp = classwise_false_positives(y_pred, y_true, weight)
    fn = classwise_false_negatives(y_pred, y_true, weight)

    dice = 2 * tp / (2 * tp + fp + fn + EPSILON)
    return dice

def classwise_mcc(y_pred, y_true, weight=None):
    """
    Computes classwise Mathews Correlation Coefficient.
    Returns: Tensor with class-wise MCC scores (C,).
    """
    tp = classwise_true_positives(y_pred, y_true, weight)
    tn = classwise_true_negatives(y_pred, y_true, weight)
    fp = classwise_false_positives(y_pred, y_true, weight)
    fn = classwise_false_negatives(y_pred, y_true, weight)

    num1 = torch.log(tp + EPSILON) + torch.log(tn + EPSILON)
    num2 = torch.log(fp + EPSILON) + torch.log(fn + EPSILON)
    
    # We use a log-sum-exp trick or similar if needed, but direct computation is usually fine for these counts
    # The original implementation used a safe log formula:
    den = 0.5 * (
        torch.log(tp + fp + EPSILON)
        + torch.log(tp + fn + EPSILON)
        + torch.log(tn + fp + EPSILON)
        + torch.log(tn + fn + EPSILON)
    )

    classwise_mcc_score = torch.exp(num1 - den) - torch.exp(num2 - den)
    return classwise_mcc_score

# -----------------------------------------------------------------------------
# Aggregated Metrics & Losses (High-Level API)
# These replace the legacy implementations.
# -----------------------------------------------------------------------------

def dice(y_pred, y_true, weight=None, axes=None):
    """
    Computes mean Dice coefficient across all classes.
    Replacement for legacy dice(). Ignores 'axes' to ensure correctness.
    """
    return torch.mean(classwise_dice(y_pred, y_true, weight))

def iou(y_pred, y_true, weight=None, axes=None):
    """
    Computes mean IoU coefficient across all classes.
    Replacement for legacy iou(). Ignores 'axes' to ensure correctness.
    """
    return torch.mean(classwise_iou(y_pred, y_true, weight))

def mcc(y_pred, y_true, weight=None, axes=None):
    """
    Computes mean MCC across all classes.
    """
    return torch.mean(classwise_mcc(y_pred, y_true, weight))


def crossentropy_loss(y_pred, y_true, weight=None, axes=None):
    """
    Computes the standard Cross Entropy Loss.
    """
    epsilon = 1e-12
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    if weight is not None:
        ce = weight * y_true * torch.log(y_pred)
        # Normalize by sum of weights
        return -torch.sum(ce) / (torch.sum(weight) + epsilon)
    else:
        ce = y_true * torch.log(y_pred)
        # Normalize by total pixels (mean reduction)
        return -torch.mean(ce) # This is equivalent to sum() / count()


def dice_loss(y_pred, y_true, weight=None, axes=None):
    """Computes Dice Loss (1 - Mean Dice)."""
    return 1 - dice(y_pred, y_true, weight)

def iou_loss(y_pred, y_true, weight=None, axes=None):
    """Computes IoU Loss (1 - Mean IoU)."""
    return 1 - iou(y_pred, y_true, weight)

def mcc_loss(y_pred, y_true, weight=None, axes=None):
    """Computes MCC Loss (1 - Mean MCC)."""
    return 1 - mcc(y_pred, y_true, weight)


def tversky_loss(y_pred, y_true, alpha=0.3, beta=0.7, weight=None):
    """
    Computes Tversky Loss (Global Mean).
    Supports alpha and beta as lists for class-wise weighting.
    """
    tp = classwise_true_positives(y_pred, y_true, weight)
    fp = classwise_false_positives(y_pred, y_true, weight)
    fn = classwise_false_negatives(y_pred, y_true, weight)
    
    # If alpha/beta are lists, convert them to tensors matching the device of tp
    if isinstance(alpha, list):
        alpha = torch.tensor(alpha, device=tp.device, dtype=tp.dtype)
    if isinstance(beta, list):
        beta = torch.tensor(beta, device=tp.device, dtype=tp.dtype)
        
    tversky_index = tp / (tp + alpha * fp + beta * fn + EPSILON)
    return 1 - torch.mean(tversky_index)


def generalized_dice_loss(y_pred, y_true, weight=None):
    """Computes the Generalized Dice Loss (GDL)."""
    tp = classwise_true_positives(y_pred, y_true, weight)
    fp = classwise_false_positives(y_pred, y_true, weight)
    fn = classwise_false_negatives(y_pred, y_true, weight)
    
    vol = tp + fn
    w_c = 1 / (vol ** 2 + EPSILON)
    
    numerator = torch.sum(w_c * tp)
    denominator = torch.sum(w_c * (2 * tp + fp + fn))
    
    gdl = 1 - 2 * numerator / (denominator + EPSILON)
    return gdl

# -----------------------------------------------------------------------------
# Weighted Losses (Using Class Importance)
# -----------------------------------------------------------------------------

def weighted_crossentropy_loss(y_pred, y_true, importance, weight=None):
    """
    Computes weighted cross-entropy loss using class importance.
    """
    epsilon = 1e-12
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    if weight is not None:
        ce = weight * y_true * torch.log(y_pred)
        # Sum over spatial dims to get per-class loss contribution?
        # Standard implementation in this codebase was:
        # dim=[0,2,3] sum, divide by counts.
        counts = torch.sum(weight, dim=[0, 2, 3])
    else:
        ce = y_true * torch.log(y_pred)
        counts = y_true.shape[0] * y_true.shape[2] * y_true.shape[3]
    
    # Per-class CE = -(sum) / count
    ce = -(torch.sum(ce, dim=[0, 2, 3])) / (counts + epsilon)
    
    # Weighted sum
    return torch.sum(ce * importance)


def weighted_mcc_loss(y_pred, y_true, importance, weight=None):
    classwise_mcc_score = classwise_mcc(y_pred, y_true, weight)
    return 1 - torch.sum(classwise_mcc_score * importance)

def weighted_iou_loss(y_pred, y_true, importance, weight=None):
    return torch.sum((1 - classwise_iou(y_pred, y_true, weight)) * importance)


# Combined losses
def dice_ce_loss(y_pred, y_true, weight=None, axes=None):
    return dice_loss(y_pred, y_true, weight) + crossentropy_loss(y_pred, y_true, weight)

def iou_ce_loss(y_pred, y_true, weight=None, axes=None):
    return iou_loss(y_pred, y_true, weight) + crossentropy_loss(y_pred, y_true, weight)

def mcc_ce_loss(y_pred, y_true, weight=None, axes=None):
    return mcc_loss(y_pred, y_true, weight) + crossentropy_loss(y_pred, y_true, weight)

def weighted_mcc_ce_loss(y_pred, y_true, importance, weight=None):
    return weighted_mcc_loss(y_pred, y_true, importance, weight) + weighted_crossentropy_loss(y_pred, y_true, importance, weight)

def weighted_iou_ce_loss(y_pred, y_true, importance, weight=None):
    return weighted_iou_loss(y_pred, y_true, importance, weight) + weighted_crossentropy_loss(y_pred, y_true, importance, weight)

# Note: Other complex losses like Focal Loss can be added here if needed, 
# reusing the helpers above.
