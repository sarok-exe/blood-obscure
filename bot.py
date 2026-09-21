#!/usr/bin/env python3
"""Telegram bot that obscures blood in images using the local tool.

Private chat: any photo or image document is processed automatically.
Groups: reply to an image with /cura (or attach a photo to the /cura message).
"""

import asyncio
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from obscure_blood import detect_blood, obscure

TOKEN = os.environ.get("CURA_BOT_TOKEN", "8961000567:AAGZ26RUaibvP1iVj-jm2TDcDAa20k0jbd4")
CONFIG_PATH = Path(__file__).resolve().parent / "obscure_config.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)

# Every processed image is saved locally for training and review:
#   bot_data/images/<stem>.png       original (lossless)
#   bot_data/masks/<stem>_mask.png   detection mask (training ground truth)
#   bot_data/results/<stem>.jpg      obscured result sent to the user
#   bot_data/meta/<stem>.json        sender / chat metadata
DATA_DIR = Path(__file__).resolve().parent / "bot_data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
RESULTS_DIR = DATA_DIR / "results"
META_DIR = DATA_DIR / "meta"
for _d in (IMAGES_DIR, MASKS_DIR, RESULTS_DIR, META_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HELP_TEXT = (
    "Send me a photo and I'll obscure any blood in it.\n\n"
    "In groups, reply to a photo with /cura (or attach a photo to the /cura "
    "message)."
)


def _save_artifacts(bgr, result_bytes: bytes, mask, meta: dict) -> None:
    """Persist original, mask, result and metadata for training/review."""
    stem = meta["stem"]
    cv2.imwrite(str(IMAGES_DIR / f"{stem}.png"), bgr)  # lossless original
    cv2.imwrite(str(MASKS_DIR / f"{stem}_mask.png"), mask)  # grayscale mask
    (RESULTS_DIR / f"{stem}.jpg").write_bytes(result_bytes)  # obscured result
    (META_DIR / f"{stem}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[saved] {stem} detected={meta.get('detected')}", flush=True)


def process_image(data: bytes, meta: dict | None = None) -> tuple[bytes, bool]:
    """Run the obscuring pipeline on raw image bytes.

    Returns (JPEG bytes, blood_detected). When no blood is found the image is
    returned unchanged (still re-encoded to JPEG). When meta is given the
    original, mask, result and metadata are saved under bot_data/.
    """
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode the image")
    mask = detect_blood(bgr, CONFIG)
    detected = int((mask > 0).sum()) > 0
    result = obscure(bgr, mask, "pixelate", CONFIG) if detected else bgr
    ok, buf = cv2.imencode(".jpg", result)
    if not ok:
        raise ValueError("could not encode the result")
    result_bytes = buf.tobytes()
    if meta is not None:
        meta["detected"] = detected
        _save_artifacts(bgr, result_bytes, mask, meta)
    return result_bytes, detected


async def _image_bytes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bytes | None:
    """Download the photo or image document attached to the current message."""
    msg = update.message
    if msg.photo:
        file = await context.bot.get_file(msg.photo[-1].file_id)
    elif msg.document:
        file = await context.bot.get_file(msg.document.file_id)
    else:
        return None
    return bytes(await file.download_as_bytearray())


async def _process_and_reply(update: Update, data: bytes, source: str) -> None:
    msg = update.message
    stem = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{msg.chat.id}_{msg.message_id}"
    meta = {
        "stem": stem,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chat_type": msg.chat.type,
        "chat_id": msg.chat.id,
        "user_id": msg.from_user.id if msg.from_user else None,
        "username": msg.from_user.username if msg.from_user else None,
        "message_id": msg.message_id,
        "source": source,
    }
    try:
        result, detected = await asyncio.to_thread(process_image, data, meta)
    except Exception as exc:  # noqa: BLE001 - reply with a short message
        await update.message.reply_text(f"Sorry, I could not process that image: {exc}")
        return
    caption = None if detected else "No blood detected"
    await update.message.reply_photo(
        photo=io.BytesIO(result), filename="result.jpg", caption=caption
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(HELP_TEXT)


async def cura(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        # e.g. an edited message or channel post with /cura - nothing to do.
        print(f"[cura] skipped: update {update.update_id} has no message")
        return
    replied = msg.reply_to_message
    print(f"[cura] chat={msg.chat.type} has_photo={bool(msg.photo)} "
          f"reply_to={replied is not None} reply_photo={bool(replied and replied.photo)} "
          f"reply_doc={bool(replied and replied.document)}")
    data = None
    source = "cura_reply"
    if msg.photo:
        # Photo attached directly to the /cura command message.
        source = "cura_attached"
        data = await _image_bytes(update, context)
    elif msg.reply_to_message is not None:
        replied = msg.reply_to_message
        if replied.photo:
            file = await context.bot.get_file(replied.photo[-1].file_id)
            data = bytes(await file.download_as_bytearray())
        elif replied.document:
            file = await context.bot.get_file(replied.document.file_id)
            data = bytes(await file.download_as_bytearray())
    if data is None:
        await msg.reply_text(
            "Reply to a photo with /cura (or attach a photo to the /cura "
            "message) to obscure blood."
        )
        return
    await _process_and_reply(update, data, source)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # In groups only /cura triggers processing; private chat auto-processes.
    if update.message is None or update.message.chat.type != "private":
        return
    data = await _image_bytes(update, context)
    if data is None:
        return
    await _process_and_reply(update, data, "private_auto")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log any handler error and keep the bot alive."""
    uid = update.update_id if isinstance(update, Update) else "?"
    print(f"[error] update={uid}: {context.error!r}", flush=True)


def main() -> None:
    app = Application.builder().token(TOKEN).build()

    async def post_init(application: Application) -> None:
        await application.bot.get_me()
        print(f"bot started as @{application.bot.username}")

    app.post_init = post_init
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cura", cura))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, on_photo))
    app.run_polling()


if __name__ == "__main__":
    main()