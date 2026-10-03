"""One classifier for Track A, one U-Net for Track B. Fixed across all imbalance runs."""
from src.utils import LESIONS, NUM_GRADES


def build_classifier(name="efficientnet_b0", pretrained=True):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=NUM_GRADES)  # 5 logits


def build_segmenter(encoder="resnet34", pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=len(LESIONS))  # 4 logits (sigmoid applied in loss/eval)


def build_model(cfg, pretrained=None):
    m = cfg["model"]
    pretrained = m["pretrained"] if pretrained is None else pretrained
    if cfg["task"] == "grading":
        return build_classifier(m["name"], pretrained)
    return build_segmenter(m["encoder"], pretrained)
