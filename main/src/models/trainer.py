# src/models/trainer.py
"""
Trainer that supports both:
- TensorFlow/Keras (Transformer)
- PyTorch (TFT)

Fixes included:
- TF load_model now passes custom_objects and handles .keras path
- PyTorch ReduceLROnPlateau 'verbose' kw removed for compatibility
- Clean train/evaluate/predict API, saving history and plots handled by main
"""

import os
import time
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow import keras

import torch
from torch.utils.data import DataLoader, TensorDataset


class ModelTrainer:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.framework = None
        self.optimizer = None
        self.criterion = None
        self.model_class = None
        self.model_params = {}

    # ---------- Common ----------
    def compile_model(self, model, learning_rate=1e-4):
        """Compile for TF or prepare for Torch."""
        # Remember how to rebuild later
        self.model_class = getattr(model, "__class__", None)
        if hasattr(model, "get_init_params"):
            self.model_params = model.get_init_params()
        else:
            self.model_params = getattr(model, "_init_params", {})

        if isinstance(model, tf.keras.Model):
            self.framework = "tf"
            opt = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=opt, loss="mse", metrics=["mae"])
        else:
            self.framework = "torch"
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            self.criterion = torch.nn.MSELoss()

    def train(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int,
        batch_size: int,
        model_save_path: str,
    ):
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        start = time.time()
        if self.framework == "tf":
            history, best_path = self._train_tf(
                model, X_train, y_train, X_val, y_val, epochs, batch_size, model_save_path
            )
        else:
            history, best_path = self._train_torch(
                model, X_train, y_train, X_val, y_val, epochs, batch_size, model_save_path
            )
        end = time.time()
        return {
            "history": history,
            "training_time": end - start,
            "epochs_trained": len(history.get("loss", [])),
            "final_val_loss": float(min(history.get("val_loss", [np.inf]))),
            "best_model_path": best_path,
        }

    def evaluate(self, model, X_test, y_test):
        if self.framework == "tf":
            out = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
            return {"mae": float(out.get("mae", np.nan)), "rmse": float(np.sqrt(out["loss"]))}
        else:
            model.eval()
            with torch.no_grad():
                # Convert numpy → tensor
                x_tensor = torch.from_numpy(X_test).float()
                y_tensor = torch.from_numpy(y_test).float().view(-1, 1)

                # Check model output type using tensor input, not numpy
                test_out = model(x_tensor[:1])
                if isinstance(test_out, tuple):
                    pred = model(x_tensor)[0]
                else:
                    pred = model(x_tensor)

                mse = torch.mean((pred - y_tensor) ** 2).item()
                mae = torch.mean(torch.abs(pred - y_tensor)).item()

            return {"mae": float(mae), "rmse": float(np.sqrt(mse))}


    def predict(self, model, X):
        if self.framework == "tf":
            return model.predict(X, verbose=0).reshape(-1, 1)
        else:
            model.eval()
            with torch.no_grad():
                x = torch.from_numpy(X).float()
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                return out.cpu().numpy().reshape(-1, 1)

    def save_results_json(self, path, payload: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    # ---------- TensorFlow ----------
    def _train_tf(self, model, X_train, y_train, X_val, y_val, epochs, batch_size, model_path):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=model_path if model_path.endswith(".keras") else "models/transformer_model.keras",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        ]
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
        ).history

        # ensure keys exist (for plotting downstream)
        for k in ["loss", "val_loss", "mae", "val_mae"]:
            hist.setdefault(k, [])
        best_path = callbacks[0].filepath
        return hist, best_path

    def load_model(self, model_path: str):
        """Load TF Keras .keras or Torch .pt model."""
        if model_path.endswith(".keras"):
            # Provide custom_objects so Keras can rebuild the custom classes
            from src.models.transformer import Transformer, PositionalEncoding, TransformerEncoderBlock  # noqa: F401

            custom_objects = {
                "Transformer": Transformer,
                "PositionalEncoding": PositionalEncoding,
                "TransformerEncoderBlock": TransformerEncoderBlock,
            }
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
            # Re-compile (optimizer state not needed for inference)
            model.compile(optimizer=optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
            self.framework = "tf"
            return model
        elif model_path.endswith(".pt"):
            import torch
            from src.models.tft import TFTSingleStep  # noqa: F401

            if self.model_class is None:
                # Best-effort default for TFT
                model = TFTSingleStep(num_features=4, d_model=128, d_hidden=256, n_heads=8, dropout=0.1)
            else:
                model = self.model_class(**self.model_params) if self.model_params else self.model_class()
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()
            self.framework = "torch"
            return model
        else:
            raise FileNotFoundError(f"Unrecognized model extension for: {model_path}")

    # ---------- PyTorch ----------
    def _train_torch(self, model, X_train, y_train, X_val, y_val, epochs, batch_size, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        pin = torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=7, min_lr=1e-7)
        best_val = float("inf")
        history = {"loss": [], "val_loss": []}
        best_path = model_path if model_path.endswith(".pt") else "models/tft_model.pt"

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=pin)
                yb = yb.to(device, non_blocking=pin).view(-1, 1)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(xb)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = self.criterion(out, yb)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device).view(-1, 1)
                    out = model(xb)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = self.criterion(out, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            history["loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if self.verbose:
                print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)

        return history, best_path
