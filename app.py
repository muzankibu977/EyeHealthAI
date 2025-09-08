import os
import io
import csv
import base64
import datetime as dt
import asyncio
from typing import List, Tuple, Optional

import flet as ft

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageEnhance
import numpy as np

from torchvision.models import vit_b_16, ViT_B_16_Weights
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# -----------------------------
# Model definitions
# -----------------------------
class Small3DCNN(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class Hybrid3DViT(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit_dim = self.vit.heads.head.in_features
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.heads = nn.Identity()
        self.cnn3d = Small3DCNN(in_ch=1)
        cnn_dim = 64
        self.classifier = nn.Sequential(
            nn.Linear(vit_dim + cnn_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, vit_img, vol3d):
        f2d = self.vit(vit_img)
        f3d = self.cnn3d(vol3d)
        return self.classifier(torch.cat([f2d, f3d], dim=1))


class ViTWrapper(nn.Module):
    def __init__(self, hybrid_model: Hybrid3DViT, fixed_vol3d: torch.Tensor):
        super().__init__()
        self.hybrid_model = hybrid_model
        self.fixed_vol3d = fixed_vol3d

    def forward(self, vit_img):
        return self.hybrid_model(vit_img, self.fixed_vol3d)


# -----------------------------
# Inference utilities
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOLUME_SIZE = 128
vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
vit_transform = vit_weights.transforms()

to_tensor_gray = T.Compose([
    T.Resize((VOLUME_SIZE, VOLUME_SIZE)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])


def make_pseudo_volume(img_pil: Image.Image, depth: int = 4) -> torch.Tensor:
    slices = [
        img_pil,
        img_pil.transpose(Image.FLIP_LEFT_RIGHT),
        img_pil.rotate(10),
        ImageEnhance.Contrast(img_pil).enhance(1.3),
    ][:depth]
    return torch.stack([to_tensor_gray(s) for s in slices], dim=1)


class ModelBundle:
    def __init__(self, ckpt_path: str = "hybrid_3dcnn_vit_model.pth"):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        if "class_names" in checkpoint:
            self.class_names = list(checkpoint["class_names"])
        else:
            self.class_names = [f"Class {i}" for i in range(checkpoint["num_classes"])]
        self.model = Hybrid3DViT(len(self.class_names)).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_with_eigencam(self, pil_img: Image.Image) -> Tuple[str, float, Image.Image]:
        vit_img = vit_transform(pil_img).unsqueeze(0).to(DEVICE)
        vol3d = make_pseudo_volume(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(vit_img, vol3d), dim=1)[0]
            idx = int(probs.argmax().item())
            conf = float(probs[idx].item())
            label = self.class_names[idx]
        wrapped = ViTWrapper(self.model, vol3d)
        target_layers = [wrapped.hybrid_model.vit.conv_proj]
        cam = EigenCAM(model=wrapped, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=vit_img)[0, :]
        rgb_img = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return label, conf, Image.fromarray(cam_img)


# -----------------------------
# UI helpers
# -----------------------------
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def typing_effect(page, text, target):
    target.value = ""
    for ch in text:
        target.value += ch
        page.update()
        await asyncio.sleep(0.1)


# -----------------------------
# Main Flet app
# -----------------------------
def main(page: ft.Page):
    page.title = "EyeHealthAI"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = "adaptive"
    page.bgcolor = "white"

    bundle = ModelBundle()
    history: List[dict] = []
    HISTORY_FILE = "prediction_history.csv"

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            reader = csv.DictReader(f)
            history = list(reader)

    uploaded_image_pil: Optional[Image.Image] = None

    # -----------------------------
    # Login page
    # -----------------------------
    welcome_text = ft.Text("", size=26, weight=ft.FontWeight.BOLD, color="#2C3E50")
    operator_id_tf = ft.TextField(label="Operator ID", prefix_icon=ft.Icons.PERSON, width=300)
    passkey_tf = ft.TextField(label="Passkey", password=True, can_reveal_password=True,
                              prefix_icon=ft.Icons.LOCK, width=300)
    login_msg = ft.Text("", color="red")

    def try_login(e):
        if operator_id_tf.value == "RTK" and passkey_tf.value == "hyperparameters123":
            page.controls.clear()
            build_main_ui()
        else:
            login_msg.value = "Invalid credentials!"
        page.update()

    def build_login_page():
        page.controls.clear()
        login_box = ft.Container(
            content=ft.Column(
                [
                    welcome_text,
                    ft.Divider(height=10, color="transparent"),
                    ft.Text("Credentials", size=20, weight=ft.FontWeight.BOLD),
                    operator_id_tf,
                    passkey_tf,
                    ft.ElevatedButton("Login", on_click=try_login),
                    login_msg,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=30,
            border_radius=15,
            border=ft.border.all(2, "#cccccc"),
            bgcolor="white",
            shadow=ft.BoxShadow(blur_radius=15, spread_radius=2, color="#888888"),
        )
        login_page = ft.Container(
            content=ft.Row([ft.Container(content=login_box)], alignment=ft.MainAxisAlignment.CENTER),
            alignment=ft.alignment.center,
            expand=True,
            bgcolor="white",
        )
        page.add(login_page)
        page.update()

    build_login_page()
    asyncio.run(typing_effect(page, "Welcome to EyeHealthAI", welcome_text))

    # -----------------------------
    # Main UI after login
    # -----------------------------
    def build_main_ui():
        nonlocal uploaded_image_pil, history

        # Title with typing effect inside page (not AppBar)
        title_text = ft.Text("", size=30, weight=ft.FontWeight.BOLD, color="#2C3E50")
        page.add(ft.Row([
            title_text,
            ft.TextButton(
                "Logout",
                icon=ft.Icons.LOGOUT,
                style=ft.ButtonStyle(
                    bgcolor={"hovered": "#ffe6e6", "": "transparent"},
                    color={"hovered": "red", "": "black"},
                ),
                on_click=lambda e: (page.controls.clear(), build_login_page())
            )
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN))
        asyncio.run(typing_effect(page, "EyeHealthAI", title_text))

        # Patient info
        name_tf = ft.TextField(label="Patient name", prefix_icon=ft.Icons.PERSON_OUTLINE)
        phone_tf = ft.TextField(label="Phone", prefix_icon=ft.Icons.PHONE)
        dob_tf = ft.TextField(label="DOB (YYYY-MM-DD)", prefix_icon=ft.Icons.CAKE)
        oct_id_tf = ft.TextField(label="OCT ID", prefix_icon=ft.Icons.BADGE)
        age_tf = ft.TextField(label="Age", read_only=True, prefix_icon=ft.Icons.CALENDAR_MONTH)
        blood_group_tf = ft.Dropdown(
            label="Blood Group",
            options=[ft.dropdown.Option(bg) for bg in ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]]
        )
        date_tf = ft.TextField(
            label="Date",
            value=dt.datetime.now().strftime("%Y-%m-%d"),
            read_only=True,
            prefix_icon=ft.Icons.EVENT
        )

        def calc_age(e):
            try:
                birth = dt.datetime.strptime(dob_tf.value, "%Y-%m-%d")
                today = dt.datetime.today()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
                age_tf.value = str(age)
            except:
                age_tf.value = ""
            page.update()
        dob_tf.on_change = calc_age

        preview_img = ft.Image(width=320, height=240)
        cam_img = ft.Image(width=320, height=240)
        pred_text = ft.Text("", size=20, weight=ft.FontWeight.BOLD)
        conf_text = ft.Text("", size=16, italic=True)
        result_container = ft.Container(
            content=ft.Column([pred_text, conf_text]),
            padding=20,
            border_radius=10,
            bgcolor="white",
            border=ft.border.all(2, "transparent"),
        )
        loading_spinner = ft.ProgressRing(visible=False)

        # Classes on right side
        class_chips = ft.Column(
            [ft.Chip(label=ft.Text(c)) for c in bundle.class_names],
            scroll="auto"
        )

        # History table
        history_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Date")),
                ft.DataColumn(ft.Text("Name")),
                ft.DataColumn(ft.Text("Phone")),
                ft.DataColumn(ft.Text("OCT ID")),
                ft.DataColumn(ft.Text("Age")),
                ft.DataColumn(ft.Text("Prediction")),
                ft.DataColumn(ft.Text("Confidence"))
            ],
            rows=[]
        )

        def refresh_history_table():
            history_table.rows.clear()
            for h in history:
                history_table.rows.append(ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(h.get("date", ""))),
                        ft.DataCell(ft.Text(h.get("name", ""))),
                        ft.DataCell(ft.Text(h.get("phone", ""))),
                        ft.DataCell(ft.Text(h.get("oct_id", ""))),
                        ft.DataCell(ft.Text(h.get("age", ""))),
                        ft.DataCell(ft.Text(h.get("pred", ""))),
                        ft.DataCell(ft.Text(f"{float(h['conf']):.2f}" if "conf" in h else "")),
                    ]
                ))
            page.update()

        refresh_history_table()

        # File picker
        def on_file_picked(e: ft.FilePickerResultEvent):
            nonlocal uploaded_image_pil
            if not e.files:
                return
            with open(e.files[0].path, "rb") as f:
                raw = f.read()
            uploaded_image_pil = Image.open(io.BytesIO(raw)).convert("RGB")
            preview_img.src_base64 = base64.b64encode(raw).decode("utf-8")
            page.update()

        picker = ft.FilePicker(on_result=on_file_picked)
        page.overlay.append(picker)

        # Prediction
        def run_prediction(e):
            nonlocal uploaded_image_pil, history
            if uploaded_image_pil is None:
                return
            loading_spinner.visible = True
            page.update()

            label, conf, cam_pil = bundle.predict_with_eigencam(uploaded_image_pil)
            pred_text.value = f"Prediction: {label}"
            conf_text.value = f"Confidence: {conf:.2f}"
            cam_img.src_base64 = pil_to_base64(cam_pil)

            if label.lower() == "normal":
                result_container.border = ft.border.all(3, "green")
                result_container.bgcolor = "#e6ffed"
            else:
                result_container.border = ft.border.all(3, "red")
                result_container.bgcolor = "#ffe6e6"

            entry = {
                "date": dt.datetime.now().strftime("%Y-%m-%d"),
                "name": name_tf.value,
                "phone": phone_tf.value,
                "oct_id": oct_id_tf.value,
                "age": age_tf.value,
                "pred": label,
                "conf": conf
            }
            history.append(entry)

            with open(HISTORY_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["date", "name", "phone", "oct_id", "age", "pred", "conf"])
                writer.writeheader()
                writer.writerows(history)

            refresh_history_table()
            loading_spinner.visible = False
            page.update()

        def download_history(e):
            if os.path.exists(HISTORY_FILE):
                page.launch_url(HISTORY_FILE)

        def delete_history(e):
            nonlocal history
            history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            refresh_history_table()

        # Layout
        page.add(
            ft.Row([name_tf, phone_tf, dob_tf]),
            ft.Row([oct_id_tf, age_tf, blood_group_tf, date_tf]),
            ft.Text("Inputs", size=18, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.ElevatedButton("Upload Image", icon=ft.Icons.UPLOAD_FILE,
                                  on_click=lambda _: picker.pick_files(file_type=ft.FilePickerFileType.IMAGE)),
                ft.ElevatedButton("Predict + Explain", icon=ft.Icons.PLAY_ARROW, on_click=run_prediction),
                loading_spinner
            ]),
            ft.Row([
                ft.Column([
                    ft.Row([preview_img, cam_img]),
                    result_container,
                    ft.Text("Prediction History", size=18, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        ft.ElevatedButton("Download History", icon=ft.Icons.DOWNLOAD, on_click=download_history),
                        ft.ElevatedButton("Clear History", icon=ft.Icons.DELETE, on_click=delete_history),
                    ]),
                    history_table
                ], expand=True),
                ft.Container(
                    content=ft.Column([ft.Text("Available Classes", size=18, weight=ft.FontWeight.BOLD), class_chips]),
                    width=200,
                    padding=10,
                    border=ft.border.all(1, "#cccccc"),
                    border_radius=10,
                )
            ])
        )


if __name__ == "__main__":
    ft.app(target=main)
