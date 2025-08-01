import torch.nn.functional as F

def kd_feature_loss(student_logits, teacher_logits, targets,
                    student_features, teacher_features,
                    alpha=0.5, temperature=2.0, beta=0.3):
    """
    Feature KD = classic KD (CE + KL) + beta * MSE(features).
    """
    # --- hard-label + soft-label KD (same as before) ---
    ce_loss = F.cross_entropy(student_logits, targets)

    T = temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

    # --- feature alignment loss ---
    feat_loss = F.mse_loss(student_features, teacher_features)

    return alpha * ce_loss + (1 - alpha) * kl_loss + beta * feat_loss
