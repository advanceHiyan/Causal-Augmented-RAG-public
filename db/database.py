"""
SQLite database — single file for all metadata.
Tables: knowledge_bases, documents, conversations, messages
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from config.settings import DATA_DIR

DB_PATH = DATA_DIR / "causal_rag.db"


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(r[1]) for r in rows}


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db() -> None:
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            kb_id       TEXT NOT NULL,
            name        TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id         TEXT PRIMARY KEY,
            kb_id      TEXT NOT NULL,
            title      TEXT NOT NULL,
            user_id    TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS messages (
            id         TEXT PRIMARY KEY,
            conv_id    TEXT NOT NULL,
            role       TEXT NOT NULL CHECK(role IN ('user','assistant')),
            content    TEXT NOT NULL,
            meta_json  TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (conv_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            student_id  TEXT UNIQUE,
            name        TEXT DEFAULT '',
            college     TEXT DEFAULT '',
            major       TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS profile_facts (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            fact_key    TEXT NOT NULL,
            fact_value  TEXT NOT NULL,
            source      TEXT DEFAULT 'manual',
            confidence  REAL DEFAULT 1.0,
            updated_at  TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, fact_key)
        );

        CREATE TABLE IF NOT EXISTS student_state_libraries (
            id          TEXT PRIMARY KEY,
            kb_id       TEXT NOT NULL,
            name        TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            UNIQUE(kb_id, name)
        );

        CREATE TABLE IF NOT EXISTS student_kb_documents (
            id          TEXT PRIMARY KEY,
            kb_id       TEXT NOT NULL,
            state_lib_id TEXT,
            name        TEXT NOT NULL,
            file_ext    TEXT DEFAULT '',
            parsed_json TEXT DEFAULT '{}',
            raw_text    TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            FOREIGN KEY (state_lib_id) REFERENCES student_state_libraries(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_kb ON conversations(kb_id);
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id);
        CREATE INDEX IF NOT EXISTS idx_profile_facts_user ON profile_facts(user_id);
        CREATE INDEX IF NOT EXISTS idx_student_state_libs_kb ON student_state_libraries(kb_id);
        CREATE INDEX IF NOT EXISTS idx_student_docs_kb ON student_kb_documents(kb_id);
        """)

        # Backward-compatible migration for old DB files.
        conv_cols = _table_columns(conn, "conversations")
        if "user_id" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")
        student_doc_cols = _table_columns(conn, "student_kb_documents")
        if "state_lib_id" not in student_doc_cols:
            conn.execute("ALTER TABLE student_kb_documents ADD COLUMN state_lib_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_student_docs_state_lib ON student_kb_documents(state_lib_id)")
        legacy_kb_rows = conn.execute(
            """
            SELECT DISTINCT kb_id FROM student_kb_documents
            WHERE state_lib_id IS NULL OR state_lib_id=''
            """
        ).fetchall()
        for row in legacy_kb_rows:
            kb_id = str(row["kb_id"])
            lib = conn.execute(
                """
                SELECT * FROM student_state_libraries
                WHERE kb_id=? AND name=?
                """,
                (kb_id, "默认学生状态库"),
            ).fetchone()
            if lib:
                lib_id = str(lib["id"])
            else:
                lib_id = _new_id()
                now = _now()
                conn.execute(
                    """
                    INSERT INTO student_state_libraries
                    (id,kb_id,name,description,created_at,updated_at)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (lib_id, kb_id, "默认学生状态库", "迁移旧学生状态文件", now, now),
                )
            conn.execute(
                """
                UPDATE student_kb_documents
                SET state_lib_id=?
                WHERE kb_id=? AND (state_lib_id IS NULL OR state_lib_id='')
                """,
                (lib_id, kb_id),
            )


# ---------------------------------------------------------------------------
# Knowledge Base CRUD
# ---------------------------------------------------------------------------

def kb_create(name: str, description: str = "") -> dict:
    now = _now()
    kb_id = _new_id()
    with db() as conn:
        conn.execute(
            "INSERT INTO knowledge_bases (id,name,description,created_at,updated_at) VALUES (?,?,?,?,?)",
            (kb_id, name.strip(), description.strip(), now, now),
        )
    return kb_get(kb_id)


def kb_get(kb_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
    return dict(row) if row else None


def kb_get_by_name(name: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM knowledge_bases WHERE name=?", (name,)).fetchone()
    return dict(row) if row else None


def kb_list() -> list[dict]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM knowledge_bases ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def kb_delete(kb_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM knowledge_bases WHERE id=?", (kb_id,))


def kb_update(kb_id: str, name: str | None = None, description: str | None = None) -> None:
    updates, params = [], []
    if name is not None:
        updates.append("name=?"); params.append(name.strip())
    if description is not None:
        updates.append("description=?"); params.append(description.strip())
    if not updates:
        return
    updates.append("updated_at=?"); params.append(_now())
    params.append(kb_id)
    with db() as conn:
        conn.execute(f"UPDATE knowledge_bases SET {','.join(updates)} WHERE id=?", params)


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------

def doc_create(kb_id: str, name: str, chunk_count: int = 0) -> dict:
    doc_id = _new_id()
    now = _now()
    with db() as conn:
        conn.execute(
            "INSERT INTO documents (id,kb_id,name,chunk_count,created_at) VALUES (?,?,?,?,?)",
            (doc_id, kb_id, name.strip(), chunk_count, now),
        )
    return doc_get(doc_id)


def doc_get(doc_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    return dict(row) if row else None


def doc_list(kb_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE kb_id=? ORDER BY created_at DESC", (kb_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def doc_delete(doc_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------

def conv_create(kb_id: str, title: str, user_id: str | None = None) -> dict:
    conv_id = _new_id()
    now = _now()
    with db() as conn:
        conn.execute(
            "INSERT INTO conversations (id,kb_id,title,user_id,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            (conv_id, kb_id, title.strip(), user_id, now, now),
        )
    return conv_get(conv_id)


def conv_get(conv_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
    return dict(row) if row else None


def conv_list(kb_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE kb_id=? ORDER BY updated_at DESC", (kb_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def conv_delete(conv_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))


def conv_rename(conv_id: str, title: str) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE conversations SET title=?,updated_at=? WHERE id=?",
            (title.strip(), _now(), conv_id),
        )


def conv_touch(conv_id: str) -> None:
    """Update updated_at timestamp."""
    with db() as conn:
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conv_id))


def conv_bind_user(conv_id: str, user_id: str | None) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE conversations SET user_id=?, updated_at=? WHERE id=?",
            (user_id, _now(), conv_id),
        )


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

def msg_create(conv_id: str, role: str, content: str, meta: dict | None = None) -> dict:
    msg_id = _new_id()
    now = _now()
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (id,conv_id,role,content,meta_json,created_at) VALUES (?,?,?,?,?,?)",
            (msg_id, conv_id, role, content, meta_json, now),
        )
    conv_touch(conv_id)
    return msg_get(msg_id)


def msg_get(msg_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM messages WHERE id=?", (msg_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["meta"] = json.loads(d.get("meta_json") or "{}")
    return d


def msg_list(conv_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conv_id=? ORDER BY created_at ASC", (conv_id,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["meta"] = json.loads(d.get("meta_json") or "{}")
        result.append(d)
    return result


def msg_update(msg_id: str, content: str) -> None:
    with db() as conn:
        conn.execute("UPDATE messages SET content=? WHERE id=?", (content, msg_id))


def msg_delete(msg_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE id=?", (msg_id,))


# ---------------------------------------------------------------------------
# User/Profile CRUD
# ---------------------------------------------------------------------------

def user_create(
    *,
    student_id: str = "",
    name: str = "",
    college: str = "",
    major: str = "",
) -> dict:
    user_id = _new_id()
    now = _now()
    with db() as conn:
        conn.execute(
            """INSERT INTO users
               (id,student_id,name,college,major,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                user_id,
                (student_id or "").strip() or None,
                (name or "").strip(),
                (college or "").strip(),
                (major or "").strip(),
                now,
                now,
            ),
        )
    return user_get(user_id) or {}


def user_get(user_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def user_get_by_student_id(student_id: str) -> dict | None:
    sid = (student_id or "").strip()
    if not sid:
        return None
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE student_id=?", (sid,)).fetchone()
    return dict(row) if row else None


def user_list(limit: int = 100) -> list[dict]:
    lim = max(1, int(limit))
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY updated_at DESC LIMIT ?",
            (lim,),
        ).fetchall()
    return [dict(r) for r in rows]


def user_update(
    user_id: str,
    *,
    student_id: str | None = None,
    name: str | None = None,
    college: str | None = None,
    major: str | None = None,
) -> None:
    updates: list[str] = []
    params: list[Any] = []
    if student_id is not None:
        v = (student_id or "").strip()
        updates.append("student_id=?")
        params.append(v or None)
    if name is not None:
        updates.append("name=?")
        params.append((name or "").strip())
    if college is not None:
        updates.append("college=?")
        params.append((college or "").strip())
    if major is not None:
        updates.append("major=?")
        params.append((major or "").strip())
    if not updates:
        return
    updates.append("updated_at=?")
    params.append(_now())
    params.append(user_id)
    with db() as conn:
        conn.execute(f"UPDATE users SET {','.join(updates)} WHERE id=?", params)


def user_touch(user_id: str) -> None:
    if not user_id:
        return
    with db() as conn:
        conn.execute("UPDATE users SET updated_at=? WHERE id=?", (_now(), user_id))


def user_upsert(
    *,
    student_id: str = "",
    name: str = "",
    college: str = "",
    major: str = "",
) -> dict:
    sid = (student_id or "").strip()
    if sid:
        existing = user_get_by_student_id(sid)
        if existing:
            user_update(
                existing["id"],
                student_id=sid,
                name=name or existing.get("name", ""),
                college=college or existing.get("college", ""),
                major=major or existing.get("major", ""),
            )
            return user_get(existing["id"]) or existing
    return user_create(student_id=sid, name=name, college=college, major=major)


def profile_fact_upsert(
    user_id: str,
    fact_key: str,
    fact_value: str,
    *,
    source: str = "manual",
    confidence: float = 1.0,
) -> None:
    key = (fact_key or "").strip()
    value = (fact_value or "").strip()
    if not user_id or not key or not value:
        return
    now = _now()
    fact_id = _new_id()
    with db() as conn:
        conn.execute(
            """
            INSERT INTO profile_facts
            (id,user_id,fact_key,fact_value,source,confidence,updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(user_id,fact_key) DO UPDATE SET
                fact_value=excluded.fact_value,
                source=excluded.source,
                confidence=excluded.confidence,
                updated_at=excluded.updated_at
            """,
            (fact_id, user_id, key, value, (source or "manual").strip(), float(confidence), now),
        )
    user_touch(user_id)


def profile_fact_bulk_upsert(
    user_id: str,
    facts: dict[str, str],
    *,
    source: str = "llm_extract",
    confidence: float = 0.7,
) -> int:
    if not user_id or not facts:
        return 0
    n = 0
    for k, v in facts.items():
        if str(k).strip() and str(v).strip():
            profile_fact_upsert(
                user_id=user_id,
                fact_key=str(k).strip(),
                fact_value=str(v).strip(),
                source=source,
                confidence=confidence,
            )
            n += 1
    return n


def profile_fact_list(user_id: str) -> list[dict]:
    if not user_id:
        return []
    with db() as conn:
        rows = conn.execute(
            """
            SELECT user_id,fact_key,fact_value,source,confidence,updated_at
            FROM profile_facts
            WHERE user_id=?
            ORDER BY fact_key ASC
            """,
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def profile_fact_get_many(user_id: str, fact_keys: list[str]) -> dict[str, str]:
    keys = [str(k).strip() for k in (fact_keys or []) if str(k).strip()]
    if not user_id or not keys:
        return {}
    placeholders = ",".join(["?"] * len(keys))
    params: list[Any] = [user_id]
    params.extend(keys)
    with db() as conn:
        rows = conn.execute(
            f"""
            SELECT fact_key, fact_value
            FROM profile_facts
            WHERE user_id=? AND fact_key IN ({placeholders})
            """,
            params,
        ).fetchall()
    return {str(r["fact_key"]): str(r["fact_value"]) for r in rows}


def profile_fact_delete(user_id: str, fact_key: str) -> None:
    key = (fact_key or "").strip()
    if not user_id or not key:
        return
    with db() as conn:
        conn.execute(
            "DELETE FROM profile_facts WHERE user_id=? AND fact_key=?",
            (user_id, key),
        )
    user_touch(user_id)


# ---------------------------------------------------------------------------
# Student state libraries and documents (file-based, no chunking)
# ---------------------------------------------------------------------------

def student_state_lib_create(kb_id: str, name: str, description: str = "") -> dict:
    lib_id = _new_id()
    now = _now()
    with db() as conn:
        conn.execute(
            """
            INSERT INTO student_state_libraries
            (id,kb_id,name,description,created_at,updated_at)
            VALUES (?,?,?,?,?,?)
            """,
            (
                lib_id,
                kb_id,
                (name or "").strip() or f"学生状态库_{lib_id[:8]}",
                (description or "").strip(),
                now,
                now,
            ),
        )
    return student_state_lib_get(lib_id) or {}


def student_state_lib_get(lib_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM student_state_libraries WHERE id=?", (lib_id,)).fetchone()
    return dict(row) if row else None


def student_state_lib_list(kb_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT l.*,
                   (SELECT COUNT(*) FROM student_kb_documents d WHERE d.state_lib_id=l.id) AS doc_count
            FROM student_state_libraries l
            WHERE l.kb_id=?
            ORDER BY l.updated_at DESC, l.created_at DESC
            """,
            (kb_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def student_state_lib_ensure_default(kb_id: str) -> dict:
    libs = student_state_lib_list(kb_id)
    if libs:
        return libs[0]
    return student_state_lib_create(kb_id, "默认学生状态库", "学生状态文件默认分组")


def student_state_lib_update(
    lib_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
) -> None:
    updates: list[str] = []
    params: list[Any] = []
    if name is not None:
        updates.append("name=?")
        params.append((name or "").strip())
    if description is not None:
        updates.append("description=?")
        params.append((description or "").strip())
    if not updates:
        return
    updates.append("updated_at=?")
    params.append(_now())
    params.append(lib_id)
    with db() as conn:
        conn.execute(f"UPDATE student_state_libraries SET {','.join(updates)} WHERE id=?", params)


def student_state_lib_delete(lib_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM student_state_libraries WHERE id=?", (lib_id,))


def student_doc_create(
    kb_id: str,
    name: str,
    *,
    state_lib_id: str = "",
    file_ext: str = "",
    parsed: dict | None = None,
    raw_text: str = "",
) -> dict:
    doc_id = _new_id()
    now = _now()
    parsed_json = json.dumps(parsed or {}, ensure_ascii=False)
    with db() as conn:
        conn.execute(
            """
            INSERT INTO student_kb_documents
            (id,kb_id,state_lib_id,name,file_ext,parsed_json,raw_text,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                doc_id,
                kb_id,
                (state_lib_id or "").strip() or None,
                (name or "").strip() or f"student_doc_{doc_id[:8]}",
                (file_ext or "").strip().lower(),
                parsed_json,
                raw_text or "",
                now,
                now,
            ),
        )
    return student_doc_get(doc_id) or {}


def student_doc_get(doc_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM student_kb_documents WHERE id=?", (doc_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["parsed"] = json.loads(d.get("parsed_json") or "{}")
    except Exception:
        d["parsed"] = {}
    return d


def student_doc_list(kb_id: str, state_lib_id: str = "") -> list[dict]:
    params: list[Any] = [kb_id]
    where = "WHERE kb_id=?"
    if state_lib_id:
        where += " AND state_lib_id=?"
        params.append(state_lib_id)
    with db() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM student_kb_documents
            {where}
            ORDER BY updated_at DESC, created_at DESC
            """,
            params,
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        try:
            d["parsed"] = json.loads(d.get("parsed_json") or "{}")
        except Exception:
            d["parsed"] = {}
        out.append(d)
    return out


def student_doc_update(
    doc_id: str,
    *,
    name: str | None = None,
    parsed: dict | None = None,
    raw_text: str | None = None,
) -> None:
    updates: list[str] = []
    params: list[Any] = []
    if name is not None:
        updates.append("name=?")
        params.append((name or "").strip())
    if parsed is not None:
        updates.append("parsed_json=?")
        params.append(json.dumps(parsed, ensure_ascii=False))
    if raw_text is not None:
        updates.append("raw_text=?")
        params.append(raw_text)
    if not updates:
        return
    updates.append("updated_at=?")
    params.append(_now())
    params.append(doc_id)
    with db() as conn:
        conn.execute(f"UPDATE student_kb_documents SET {','.join(updates)} WHERE id=?", params)


def student_doc_delete(doc_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM student_kb_documents WHERE id=?", (doc_id,))


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def kb_stats(kb_id: str) -> dict:
    with db() as conn:
        doc_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE kb_id=?", (kb_id,)
        ).fetchone()[0]
        total_chunks = conn.execute(
            "SELECT COALESCE(SUM(chunk_count),0) FROM documents WHERE kb_id=?", (kb_id,)
        ).fetchone()[0]
        conv_count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE kb_id=?", (kb_id,)
        ).fetchone()[0]
        msg_count = conn.execute(
            """SELECT COUNT(*) FROM messages m
               JOIN conversations c ON m.conv_id=c.id
               WHERE c.kb_id=?""",
            (kb_id,),
        ).fetchone()[0]
    return {
        "doc_count": doc_count,
        "total_chunks": total_chunks,
        "conv_count": conv_count,
        "msg_count": msg_count,
    }


# Initialize on import
init_db()
