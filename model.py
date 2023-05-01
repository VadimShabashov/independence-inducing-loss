import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import defaultdict
from torch.nn.functional import normalize
from online_triplet_loss.losses import batch_hard_triplet_loss

from losses.zero_loss import ZeroLoss


class Model(pl.LightningModule):
    def __init__(self, model, device, num_unfrozen_layers, independence_loss, regularization_loss,
                 classification_loss, num_classes, embedding_dim, margin):
        super().__init__()

        # Pretrained model
        self.model = get_model(model, embedding_dim, num_unfrozen_layers)

        # Create classification layer
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Save losses
        self.independence_loss = independence_loss
        self.regularization_loss = regularization_loss

        if classification_loss == 'Enable':
            self.classification_loss = nn.CrossEntropyLoss()
        else:
            self.classification_loss = ZeroLoss(device)

        # Save margin for ranking loss
        self.margin = margin

        # Save outputs of training step
        self.training_step_outputs = defaultdict(list)

        # Get logger
        self.app_logger = logging.getLogger()

    def get_embeddings(self, x):
        # Find embeddings and perform L2 normalization
        # It's important to note, that normalization + linear layer is linear transform again
        return normalize(self.model(x), p=2)

    def get_logits(self, embeddings):
        return self.classifier(self.relu(embeddings))

    def get_predictions(self, logits):
        probs = self.softmax(logits)
        return torch.argmax(probs, dim=1)

    def forward(self, x):
        raise Exception("Use 'get_embeddings', 'get_logits' or 'get_predictions'")

    def training_step(self, batch, batch_idx):
        # Get batch data
        x, labels = batch

        # Get embeddings, logits and probabilities of classes
        embeddings = self.get_embeddings(x)
        logits = self.get_logits(embeddings)
        preds = self.get_predictions(logits)

        # Find losses
        ranking_loss = batch_hard_triplet_loss(labels, embeddings, margin=self.margin)
        regularization_loss = self.regularization_loss(embeddings)
        independence_loss = self.independence_loss(embeddings)
        classification_loss = self.classification_loss(logits, labels)
        loss = ranking_loss + regularization_loss + independence_loss + classification_loss

        # Find accuracy
        accuracy = torch.sum(torch.eq(preds, labels)) / len(preds)

        # define outputs with losses
        outputs = {
            'loss': loss,
            'ranking_loss': ranking_loss,
            'regularization_loss': regularization_loss,
            'independence_loss': independence_loss,
            'classification_loss': classification_loss,
            'accuracy': accuracy
        }

        # Save losses for finding mean in the end of epoch
        for output_name, output_value in outputs.items():
            self.training_step_outputs[output_name].append(output_value)

        return outputs

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )

        # Scheduler config
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "loss"
        }

        return [optimizer], [lr_scheduler_config]

    def on_train_epoch_end(self):
        # Calculate mean of losses for visualization
        output_means = []
        for output_name, output_values in self.training_step_outputs.items():
            output_means.append(f"{output_name} = {(sum(output_values) / len(output_values)):.2f}")

        # Find overall loss and store in logs for scheduler
        losses = self.training_step_outputs["loss"]
        self.log("loss", sum(losses) / len(losses))

        # Display losses
        self.app_logger.info(", ".join(output_means))

        # Remove values of losses for the current epoch
        self.training_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, labels = batch
        embeddings = self.get_embeddings(x)
        return embeddings, labels


def get_model(model_name, embedding_dim, num_unfrozen_layers):
    if model_name == "ViT":
        # Import model
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        # Get model
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Change last layer dimension
        model.heads.head = nn.Linear(model.heads.head.in_features, embedding_dim)

    elif model_name.startswith("ResNet"):
        if model_name == "ResNet18":
            # Import model
            from torchvision.models import resnet18, ResNet18_Weights

            # Get model
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        elif model_name == "ResNet34":
            # Import model
            from torchvision.models import resnet34, ResNet34_Weights

            # Get model
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        elif model_name == "ResNet50":
            # Import model
            from torchvision.models import resnet50, ResNet50_Weights

            # Get model
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        else:
            raise Exception(f"Unknown ResNet model {model_name}")

        # Change last layer dimension
        model.fc = nn.Linear(model.fc.in_features, embedding_dim)

    elif model_name == "AlexNet":
        # Import model
        from torchvision.models import alexnet, AlexNet_Weights

        # Get model
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        # Change last layer dimension
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, embedding_dim)

    else:
        raise Exception(f"Unknown model {model_name}")

    # Freeze required number of layers
    if isinstance(num_unfrozen_layers, int):
        for _, module in list(model.named_children())[:-num_unfrozen_layers]:
            for param in module.parameters():
                param.requires_grad = False

    return model


def run_inference(trainer, dataloader, model):
    # Inference + combine results
    embeddings, labels = zip(*trainer.predict(model, dataloaders=dataloader))
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    return embeddings, labels
