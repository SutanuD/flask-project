import os
from pathlib import Path
from flask import Blueprint, jsonify, current_app, send_from_directory, abort, request
from pydantic import BaseModel, ValidationError

from application.hyrag import chat_router
from application.knowledge import build_kb

bp = Blueprint('main', __name__)

class ChatReq(BaseModel):
    question: str
    subject: str | None = None
    grade: int | None = None

@bp.route('/', methods=['GET'])
def welcome():
    return "Welcome!"

@bp.route('/hello', methods=['GET'])
def hello():
    return "Hello World!"

@bp.route('/pdfs', methods=['GET'])
def list_pdfs():
    #pdf_dir = Path(current_app.root_path).parent / "content" / "pdfs"
    pdf_dir = Path(current_app.root_path)/ "content" / "pdfs"
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        return jsonify(error="pdf directory not found", path=str(pdf_dir)), 404

    files = sorted([p.name for p in pdf_dir.iterdir() if p. is_file() and p.suffix.lower() == '.pdf'])
    return jsonify(files=files)

@bp.route('/pdfs/<path:filename>', methods=['GET'])
def serve_pdf(filename):
    pdf_dir = Path(current_app.root_path) / "content" / "pdfs"
    # Prevent directory traversal by resolving and checking the prefix
    safe_path = (pdf_dir / filename).resolve()
    if not str(safe_path).startswith(str(pdf_dir.resolve())) or not safe_path.exists():
        abort(404)

    return send_from_directory(str(pdf_dir), filename)


@bp.route('/ingest', methods=['GET'])
def ingest_pdfs():
    #pdf_dir = Path(current_app.root_path).parent / "content" / "pdfs"
    build_kb()
    return jsonify(message="Ingestion started"), 200

@bp.route('/chat', methods=['POST'])
def chat():
    # Flask doesn't inject a typed body parameter; parse JSON and validate with Pydantic
    try:
        payload = request.get_json(force=True)
        req = ChatReq.model_validate(payload or {})
    except ValidationError as e:
        return jsonify(error="Invalid request", details=e.errors()), 400
    except Exception as e:
        return jsonify(error="Bad request", details=str(e)), 400

    ans, cites, subj, additional = chat_router(req.question, subject=req.subject, grade=req.grade)
    return jsonify(subject_detected=subj, answer=ans, citations=cites, more_info=additional)
