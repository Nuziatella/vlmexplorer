from __future__ import annotations

import torch
from PIL import Image
from PySide6.QtWidgets import QMessageBox

from app.vlm_worker import VLMWorker


class InferenceMixin:
    """Inference workflow: run, progress, chat display, cleanup."""

    def run_vlm(self) -> None:
        """Run the selected VLM on the current image and user's question."""
        prompt_text = self.get_input_text(self.prompt_input).strip()
        if not self.current_image_path or not prompt_text:
            QMessageBox.warning(self, "Error", "Please load an image and enter a question.")
            return

        if torch.cuda.is_available():
            try:
                free, _total = torch.cuda.mem_get_info()
                free_gb = free / (1024**3)
                if free_gb < 1.0:
                    resp = QMessageBox.question(
                        self,
                        "Low VRAM",
                        f"Only {free_gb:.2f} GB VRAM free. Continue anyway?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if resp != QMessageBox.Yes:
                        return
            except Exception:
                pass

        # Disable UI and show progress
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Selection and prompt
        model_id = getattr(self, "local_model_path", None) or self.model_combo.currentData()
        vision_text = self.get_input_text(self.vision_input).strip() if hasattr(self, "vision_input") else ""
        memory_text = self.get_input_text(self.memory_input).strip() if hasattr(self, "memory_input") else ""
        prompt = self.build_prompt(vision_text, memory_text, prompt_text)

        # Preloaded pipeline compatibility
        use_preloaded = (
            getattr(self, "loaded_pipeline", None) is not None
            and getattr(self, "loaded_model_name", None) == model_id
            and self.config_matches_loaded_model()
        )

        if (
            getattr(self, "loaded_pipeline", None) is not None
            and getattr(self, "loaded_model_name", None) == model_id
            and not self.config_matches_loaded_model()
        ):
            reply = QMessageBox.question(
                self,
                "Configuration Changed",
                "The loaded model was configured with different settings.\n\n"
                "Current settings (quantization, device mapping, etc.) don't match "
                "the loaded model configuration.\n\n"
                "Do you want to:\n"
                "• Continue with current loaded model (faster, but uses old settings)\n"
                "• Reload model with new settings (slower, but uses current config)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                use_preloaded = False

        gen_params = {
            "answer_top_k": self.answer_top_k.value(),
            "temperature": self.temperature.value(),
            "top_p": self.top_p.value(),
            "num_beams": self.num_beams.value(),
            "max_new_tokens": self.max_new_tokens.value(),
        }
        preprocess = {
            "use_fp16": bool(self.use_fp16.isChecked() and torch.cuda.is_available()),
            "max_image_size": int(self.max_image_size.value()),
        }
        load_opts = {
            "use_8bit": bool(self.use_8bit.isChecked()),
            "use_4bit": bool(self.use_4bit.isChecked()),
            "device_map_auto": bool(self.device_map_auto.isChecked()),
        }

        # Two-screen merge
        pil_image = None
        if self.screens_combo.currentText() in ("Two Screens", "Two"):
            if not getattr(self, "image2_path", None):
                QMessageBox.warning(self, "Error", "Please select the second image for Two screens.")
                self.set_ui_enabled(True)
                self.progress_bar.setVisible(False)
                return
            img1 = Image.open(self.current_image_path).convert("RGB")
            img2 = Image.open(self.image2_path).convert("RGB")
            w = max(img1.width, img2.width)
            h = img1.height + img2.height
            pil_image = Image.new("RGB", (w, h), (0, 0, 0))
            pil_image.paste(img1, (0, 0))
            pil_image.paste(img2, (0, img1.height))

        # Worker
        self.worker = VLMWorker(
            model_id,
            self.current_image_path,
            prompt,
            cache_dir=getattr(self, "cache_dir", None),
            gen_params=gen_params,
            preprocess=preprocess,
            pil_image=pil_image,
            load_opts=load_opts,
            task=self.task_combo.currentText(),
            preloaded_pipeline=self.loaded_pipeline if use_preloaded else None,
        )
        self.worker.finished.connect(self.on_vlm_finished)
        self.worker.error.connect(self.on_vlm_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.timing.connect(self.on_vlm_timing)
        self.worker.start()

    def update_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def get_input_text(self, widget) -> str:
        if hasattr(widget, "text"):
            return widget.text()
        return widget.toPlainText()

    def on_vlm_finished(self, result: str) -> None:
        user_message = self.get_input_text(self.prompt_input)
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": self.get_timestamp(),
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": result,
            "timestamp": self.get_timestamp(),
        })
        self.update_chat_display()
        if hasattr(self.prompt_input, "clear"):
            self.prompt_input.clear()
        else:
            self.prompt_input.setPlainText("")
        self.last_answer = result
        self.cleanup_worker()

    def get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%H:%M")

    def update_chat_display(self) -> None:
        if not hasattr(self, "chat_history"):
            return
        chat_html = ""
        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg["timestamp"]
            if role == "user":
                chat_html += f"""
                <div style="margin: 10px 0; text-align: right;">
                    <div style="background: #0084ff; color: white; padding: 8px 12px; border-radius: 18px; display: inline-block; max-width: 70%; word-wrap: break-word;">
                        {content}
                    </div>
                    <div style="font-size: 11px; color: #888; margin-top: 2px;">
                        You • {timestamp}
                    </div>
                </div>
                """
            else:
                chat_html += f"""
                <div style="margin: 10px 0; text-align: left;">
                    <div style="background: #f1f1f1; color: #333; padding: 8px 12px; border-radius: 18px; display: inline-block; max-width: 70%; word-wrap: break-word;">
                        {content}
                    </div>
                    <div style="font-size: 11px; color: #888; margin-top: 2px;">
                        Assistant • {timestamp}
                    </div>
                </div>
                """
        self.chat_history.setHtml(chat_html)
        scrollbar = self.chat_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_vlm_timing(self, seconds: float) -> None:
        self.time_label.setText(f"Time: {seconds*1000:.0f} ms")

    def on_vlm_error(self, error_msg: str) -> None:
        QMessageBox.critical(self, "Error", error_msg)
        self.cleanup_worker()

    def cleanup_worker(self) -> None:
        if getattr(self, "worker", None):
            self.worker.quit()
            self.worker.wait()
            self.worker = None
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)

    def set_ui_enabled(self, enabled: bool) -> None:
        if hasattr(self, "load_btn"):
            self.load_btn.setEnabled(enabled)
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(enabled and bool(self.current_image_path))
        if hasattr(self, "model_combo"):
            self.model_combo.setEnabled(enabled)
        if hasattr(self, "prompt_input"):
            self.prompt_input.setEnabled(enabled)
        if hasattr(self, "local_model_btn"):
            self.local_model_btn.setEnabled(enabled)
        if hasattr(self, "cache_btn"):
            self.cache_btn.setEnabled(enabled)
        if hasattr(self, "load_btn2") and hasattr(self, "screens_combo"):
            self.load_btn2.setEnabled(
                enabled and self.screens_combo.currentText() in ("Two Screens", "Two")
            )

    def build_prompt(self, vision_text: str, memory_text: str, user_question: str) -> str:
        screens_line = (
            "[Input: Two gameplay screenshots]"
            if self.screens_combo.currentText() in ("Two Screens", "Two")
            else "[Input: One gameplay screenshot]"
        )

        # Decide whether the user explicitly requested JSON output
        text_bundle = " ".join([
            str(vision_text or ""),
            str(memory_text or ""),
            str(user_question or ""),
        ]).lower()
        wants_json = any(
            key in text_bundle
            for key in [
                "json",
                "schema",
                "return a json",
                "return only json",
                "json only",
                "output json",
                "as json",
                "in json",
                "json format",
                "json-formatted",
                "strict json",
                "machine-readable json",
                "provide json",
                "provide a json",
            ]
        )

        if wants_json:
            # Structured JSON output when explicitly requested
            return (
                f"{screens_line}\n\n"
                "Task: Analyze this Pokémon screenshot and respond ONLY with a single JSON object, no extra text. "
                "Player buttons: A, B, Up, Down, Left, Right, Start, Select, L, R.\n\n"
                "UI Modes:\n"
                "- BATTLE: HP bars, Pokémon sprites, battle interface\n"
                "- DIALOG: Text boxes\n"
                "- MENU: Menu lists, Pokémon stats/items\n"
                "- OVERWORLD: Player on map, no overlays\n\n"
                "Steps:\n"
                "1) Detect visible UI elements (HP bars, text boxes, menus, map)\n"
                "2) Pick the dominant screen mode\n"
                "3) If multiple modes present, pick the one requiring immediate player action\n\n"
                "JSON Schema:\n"
                "{\n"
                " \"ui_mode\": \"OVERWORLD|DIALOG|MENU|BATTLE\",\n"
                " \"visible_text\": \"concise text\",\n"
                " \"hp_estimates\": {\"player_hp_pct\":0..100,\"opponent_hp_pct\":0..100},\n"
                " \"menu_options\": [\"list of strings\"],\n"
                " \"suggested_action\": \"A|B|UP|DOWN|LEFT|RIGHT|START|SELECT|L|R\",\n"
                " \"reason\": \"brief, <10 words\",\n"
                " \"confidence\": 0.0..1.0\n"
                "}\n\n"
                "Rules:\n"
                "- Non-BATTLE → hp_estimates = {\"player_hp_pct\":0,\"opponent_hp_pct\":0}\n"
                "- Focus on player’s next required action\n"
                "- Default ui_mode=OVERWORLD if unsure\n"
                "- Default suggested_action=\"A\" with confidence=0.0 if unsure\n\n"
                f"Context:\nVISION_PROMPT: {vision_text}\nMEMORY: {memory_text}\n\n"
                "Return completed JSON object:\n"
                "{\n"
                " \"visible_text\": \"\",\n"
                " \"hp_estimates\": {\"player_hp_pct\":0,\"opponent_hp_pct\":0},\n"
                " \"menu_options\": [],\n"
                " \"suggested_action\": \"\",\n"
                " \"reason\": \"\",\n"
                " \"ui_mode\": \"\",\n"
                " \"confidence\": 0.0\n"
                "}\n\n"
                f"User Question: {user_question}\n"
            )
        else:
            # Natural conversational output by default
            return (
                f"{screens_line}\n\n"
                "Task: Analyze this Pokémon screenshot and answer the user's question clearly and concisely. "
                "Player buttons: A, B, Up, Down, Left, Right, Start, Select, L, R.\n\n"
                "UI Modes:\n"
                "- BATTLE: HP bars, Pokémon sprites, battle interface\n"
                "- DIALOG: Text boxes\n"
                "- MENU: Menu lists, Pokémon stats/items\n"
                "- OVERWORLD: Player on map, no overlays\n\n"
                "Guidance:\n"
                "1) Identify visible UI elements (HP bars, text boxes, menus, map)\n"
                "2) Decide the current screen mode\n"
                "3) If multiple modes present, prefer the one requiring immediate action\n\n"
                f"Context:\nVISION_PROMPT: {vision_text}\nMEMORY: {memory_text}\n\n"
                f"User Question: {user_question}\n"
            )
