from vit_pytorch import SimpleViT

def get_vit_model():
    v = SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=128
    )
    return v