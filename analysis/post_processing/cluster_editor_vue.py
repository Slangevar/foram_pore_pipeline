"""
3D Cluster Editor – Vue.js + Vuetify Edition.

Drop-in replacement for cluster_editor.py with a modern Material Design UI.
Same Python backend, same data format, same launch command.

Usage:
    python analysis/post_processing/cluster_editor_vue.py data/samples/editor_state --port 5010
"""

import numpy as np
import json
import os
import sys
import glob
import argparse
import errno
import pwd
import stat
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import mimetypes

# ── State ──
STATE = {}
VUE_VERSION = os.environ.get('CLUSTER_EDITOR_VUE_VERSION', '3.3.9')
THREE_VERSION = os.environ.get('CLUSTER_EDITOR_THREE_VERSION', '0.164.0')


def display_name_from_npz(path):
    base = os.path.basename(path)
    if '_cluster_state' in base:
        return base.split('_cluster_state', 1)[0]
    return base.replace('.npz', '')


def is_editor_state_npz(path):
    try:
        d = np.load(path, allow_pickle=True)
        needed = {'centroids_3d', 'labels', 'labeled_pores'}
        return needed.issubset(set(d.keys()))
    except Exception:
        return False


def prefer_manual_review_states(npz_files, folder):
    """Load corrected manual-review states on startup when they exist."""
    parent_dir = os.path.dirname(os.path.abspath(folder))
    if os.path.basename(parent_dir) == 'manual_review':
        return npz_files, []

    manual_state_dir = os.path.join(parent_dir, 'manual_review', 'editor_state')
    if not os.path.isdir(manual_state_dir):
        return npz_files, []

    manual_by_name = {}
    for path in glob.glob(os.path.join(manual_state_dir, '*.npz')):
        if is_editor_state_npz(path):
            manual_by_name[display_name_from_npz(path)] = os.path.abspath(path)

    if not manual_by_name:
        return npz_files, []

    resolved_files = []
    corrected_names = []
    for path in npz_files:
        name = display_name_from_npz(path)
        manual_path = manual_by_name.get(name)
        if manual_path:
            resolved_files.append(manual_path)
            corrected_names.append(name)
        else:
            resolved_files.append(path)

    return resolved_files, corrected_names


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    result = {
        'labels': d['labels'].copy(),
        'centroids': d['centroids_3d'],
        'labeled_pores': d.get('labeled_pores', None),
        'shell_mask': d.get('shell_mask', None),
        'volume_path': str(d.get('volume_path', '')),
        'state_path': path,
    }
    result['pore_voxels'] = d.get('pore_voxels', None)
    result['pore_voxels_owner'] = d.get('pore_voxels_owner', None)
    result['chamber_voxels'] = d.get('chamber_voxels', None)
    return result


def build_html(npz_files, initial_idx=0, corrected_names=None):
    STATE['npz_files'] = npz_files
    corrected_names = corrected_names or []
    corrected_set = set(corrected_names)
    volumes = []
    for i, f in enumerate(npz_files):
        name = display_name_from_npz(f)
        volumes.append({
            'idx': i,
            'name': name,
            'file': f,
            'corrected': name in corrected_set,
        })
    STATE['volumes'] = volumes
    data_json = json.dumps(volumes)
    html = (HTML_TEMPLATE
            .replace('__DATA__', data_json)
            .replace('__INITIAL__', str(initial_idx))
            .replace('__VUE_VERSION__', VUE_VERSION)
            .replace('__THREE_VERSION__', THREE_VERSION))
    STATE['html_bytes'] = html.encode('utf-8')
    print(f"Loaded metadata for {len(volumes)} volumes.")
    return html


def _stem_from_volume_path(volume_path):
    name = os.path.basename(volume_path)
    for suffix in ('_pores_cleaned.npy', '_cleaned.npy', '_pred.npy', '.npy'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _project_root_from_state_path(npz_path):
    abs_path = os.path.abspath(npz_path)
    parts = abs_path.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == 'data':
            prefix = os.sep.join(parts[:i])
            return prefix or os.sep
    return os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))


def _project_relative_path(path):
    parts = os.path.normpath(path).split(os.sep)
    for i, part in enumerate(parts):
        if part == 'data':
            return os.path.join(*parts[i:])
    return None


def _relative_to_project(path, project_root):
    try:
        rel = os.path.relpath(path, project_root)
    except ValueError:
        return path
    if rel == os.curdir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return path
    return rel


def _existing_volume_candidate(volume_path, npz_path):
    candidates = [volume_path]
    if npz_path:
        project_root = _project_root_from_state_path(npz_path)
        if os.path.isabs(volume_path):
            project_rel = _project_relative_path(volume_path)
            if project_rel:
                candidates.append(os.path.join(project_root, project_rel))
        else:
            candidates.append(os.path.join(project_root, volume_path))

    seen = set()
    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if abs_candidate in seen:
            continue
        seen.add(abs_candidate)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def resolve_volume_path(volume_path, npz_path=None):
    """Resolve source volumes from the state file, independent of launch cwd."""
    volume_path = os.path.expanduser(os.path.expandvars(str(volume_path)))
    found = _existing_volume_candidate(volume_path, npz_path)
    if found:
        return found

    old_prefix = os.path.join('data', 'pores_cleaned') + os.sep
    new_prefix = os.path.join('data', 'final_foram_state', 'predicted_cleaned_volume') + os.sep
    if volume_path.startswith(old_prefix):
        migrated_path = new_prefix + volume_path[len(old_prefix):]
        found = _existing_volume_candidate(migrated_path, npz_path)
        if found:
            return found

    raise FileNotFoundError(
        f"Could not find source volume '{volume_path}'. "
        f"Tried the path as given and, for relative paths, relative to the "
        f"project root inferred from the editor state. If this is an older "
        f"editor state, update volume_path from '{old_prefix}' to '{new_prefix}'."
    )


def _make_group_writable(path):
    try:
        mode = os.stat(path).st_mode
        mode |= stat.S_IRGRP | stat.S_IWGRP
        if os.path.isdir(path):
            mode |= stat.S_IXGRP | stat.S_ISGID
        os.chmod(path, mode)
    except OSError:
        pass


def _temp_output_path(path):
    parent = os.path.dirname(path)
    name = os.path.basename(path)
    return os.path.join(parent, f'.{name}.tmp.{os.getpid()}')


def _replace_with_temp(path, writer):
    tmp_path = _temp_output_path(path)
    try:
        writer(tmp_path)
        _make_group_writable(tmp_path)
        os.replace(tmp_path, path)
        _make_group_writable(path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _save_npy_replace(path, array):
    def writer(tmp_path):
        with open(tmp_path, 'wb') as f:
            np.save(f, array)
    _replace_with_temp(path, writer)


def _savez_compressed_replace(path, **arrays):
    def writer(tmp_path):
        with open(tmp_path, 'wb') as f:
            np.savez_compressed(f, **arrays)
    _replace_with_temp(path, writer)


def _write_text_replace(path, text):
    def writer(tmp_path):
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(text)
    _replace_with_temp(path, writer)


def save_corrected(npz_path, new_labels):
    npz_path = os.path.abspath(npz_path)
    project_root = _project_root_from_state_path(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    labeled_pores = d['labeled_pores']
    volume_path = resolve_volume_path(str(d['volume_path']), npz_path)
    labels = np.array(new_labels)
    old_labels = np.array(d['labels'])

    original = np.load(volume_path)
    vol = original[0] if original.ndim == 4 else original
    output = vol.copy()
    # Rebuild chamber labels from the editor mapping every time.  This matters
    # when saving an already-corrected volume, whose pore voxels may be labels
    # 2, 3, 4, ... rather than the original undifferentiated pore label 2.
    output[labeled_pores > 0] = 0

    for cid in sorted(set(labels)):
        if cid < 2:
            continue  # label 0 = deleted pore; skip so it remains background
        pi = np.where(labels == cid)[0] + 1
        output[np.isin(labeled_pores, pi)] = cid

    parent_dir = os.path.dirname(npz_path)
    root_dir = os.path.dirname(parent_dir)
    if os.path.basename(root_dir) == 'manual_review':
        manual_root = root_dir
    else:
        manual_root = os.path.join(root_dir, 'manual_review')

    os.makedirs(manual_root, exist_ok=True)
    _make_group_writable(manual_root)

    out_dir = os.path.join(manual_root, 'corrected_volume')
    state_out_dir = os.path.join(manual_root, 'editor_state')
    summary_npz_dir = os.path.join(manual_root, 'summary_npz')
    summary_tsv_dir = os.path.join(manual_root, 'summary_tsv')
    for p in (out_dir, state_out_dir, summary_npz_dir, summary_tsv_dir):
        os.makedirs(p, exist_ok=True)
        _make_group_writable(p)

    base = display_name_from_npz(npz_path) or _stem_from_volume_path(volume_path)

    out_path = os.path.join(out_dir, f'{base}.npy')
    _save_npy_replace(out_path, output)

    state_out_path = os.path.join(state_out_dir, f'{base}.npz')
    _savez_compressed_replace(
        state_out_path,
        centroids_3d=d['centroids_3d'],
        labels=labels,
        labeled_pores=d['labeled_pores'],
        shell_mask=d['shell_mask'],
        volume_path=np.array(_relative_to_project(out_path, project_root)),
        pore_voxels=d['pore_voxels'],
        pore_voxels_owner=d['pore_voxels_owner'],
        chamber_voxels=d['chamber_voxels']
    )

    changed_pores = int(np.sum(old_labels != labels))
    chambers_before = int(np.unique(old_labels[old_labels >= 2]).size)
    chambers_after = int(np.unique(labels[labels >= 2]).size)
    deleted_pores = int(np.sum(labels < 2))
    edited_at = datetime.now().isoformat(timespec='seconds')

    summary_npz_path = os.path.join(summary_npz_dir, f'{base}.npz')
    _savez_compressed_replace(
        summary_npz_path,
        base_name=np.array(base),
        source_state_path=np.array(npz_path),
        corrected_state_path=np.array(state_out_path),
        source_volume_path=np.array(volume_path),
        corrected_volume_path=np.array(out_path),
        total_pores=np.array(labels.size),
        deleted_pores=np.array(deleted_pores),
        changed_pores=np.array(changed_pores),
        chambers_before=np.array(chambers_before),
        chambers_after=np.array(chambers_after),
        edited_at=np.array(edited_at),
    )

    summary_tsv_path = os.path.join(summary_tsv_dir, f'{base}.tsv')
    summary_tsv = (
        'volume\ttotal_pores\tdeleted_pores\tchanged_pores\tchambers_before\tchambers_after\tedited_at\n'
        f'{base}\t{labels.size}\t{deleted_pores}\t{changed_pores}\t{chambers_before}\t{chambers_after}\t{edited_at}\n'
    )
    _write_text_replace(summary_tsv_path, summary_tsv)

    return {
        'saved_volume': out_path,
        'saved_state': state_out_path,
        'summary_npz': summary_npz_path,
        'summary_tsv': summary_tsv_path,
        'manual_root': manual_root,
    }


class EditorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            data = STATE['html_bytes']
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/api/volumes':
            payload = json.dumps(STATE['volumes']).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path.startswith('/api/volume/'):
            try:
                idx = int(self.path.split('/')[-1])
                f = STATE['npz_files'][idx]
                d = load_npz(f)
                vol_data = {
                    'name': display_name_from_npz(f),
                    'file': f,
                    'centroids': d['centroids'].tolist(),
                    'labels': d['labels'].tolist(),
                    'pore_voxels': d['pore_voxels'].tolist() if d.get('pore_voxels') is not None else [],
                    'pore_voxels_owner': d['pore_voxels_owner'].tolist() if d.get('pore_voxels_owner') is not None else [],
                    'chamber_voxels': d['chamber_voxels'].tolist() if d.get('chamber_voxels') is not None else []
                }
                payload = json.dumps(vol_data).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                self.send_error(500, str(e))
        elif self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
        elif self.path.startswith('/static/'):
            rel_path = self.path[len('/static/'):].split('?', 1)[0]
            static_dir = STATE.get('static_dir')
            if not static_dir:
                self.send_error(404)
                return
            file_path = os.path.abspath(os.path.join(static_dir, rel_path))
            if not file_path.startswith(os.path.abspath(static_dir) + os.sep) or not os.path.isfile(file_path):
                self.send_error(404)
                return
            mime, _ = mimetypes.guess_type(file_path)
            mime = mime or 'application/octet-stream'
            with open(file_path, 'rb') as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_HEAD(self):
        # Needed so urlExists() in the browser can probe /static/ files
        if self.path.startswith('/static/'):
            rel_path = self.path[len('/static/'):].split('?', 1)[0]
            static_dir = STATE.get('static_dir')
            if static_dir:
                file_path = os.path.abspath(os.path.join(static_dir, rel_path))
                if file_path.startswith(os.path.abspath(static_dir) + os.sep) and os.path.isfile(file_path):
                    self.send_response(200)
                    self.end_headers()
                    return
        self.send_error(404)

    def do_POST(self):
        if self.path == '/save':
            length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(length))
            npz_path = body['file']
            labels = body['labels']
            try:
                out = save_corrected(npz_path, labels)
            except Exception as e:
                self.send_error(500, str(e))
                print(f"Save failed for {npz_path}: {e}", file=sys.stderr)
                return

            # Keep review sequential: once saved, this index loads corrected editor state.
            for i, current in enumerate(STATE.get('npz_files', [])):
                if current == npz_path:
                    STATE['npz_files'][i] = out['saved_state']
                    if i < len(STATE.get('volumes', [])):
                        STATE['volumes'][i]['file'] = out['saved_state']
                        STATE['volumes'][i]['corrected'] = True
                    break

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())
            print(f"Saved corrected state: {out['saved_state']}")
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          fmt % args))


# ── HTML Template (Vue 3 + Vuetify 3 + Three.js) ──
HTML_TEMPLATE = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>3D Cluster Editor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
:root {
    --toolbar-bg: #ffffff;
    --surface: #f6f8fb;
    --surface-2: #eef2f7;
    --text-strong: #101828;
    --text-soft: #475467;
    --stroke: #d0d5dd;
    --btn-radius: 10px;
    --primary: #175cd3;
    --primary-hover: #1849a9;
    --danger: #b42318;
    --success: #027a48;
    --legend-width: 300px;
}
html, body {
    height:100%;
    overflow:hidden;
    font-family:'Inter', 'Segoe UI', sans-serif;
    background: #eef2f7;
}
[v-cloak] { display: none; }

/* ── Toolbar (plain flexbox, like the original) ── */
#toolbar {
    position:fixed; top:0; left:0; right:0; z-index:100;
    min-height:58px; display:flex; align-items:center; gap:8px;
    padding:8px 12px;
    background: var(--toolbar-bg);
    border-bottom: 1px solid var(--stroke);
    font-size:13px; color:var(--text-strong);
}
#toolbar .title {
    font-weight:600;
    letter-spacing:0.2px;
    font-size:14px;
    margin-right:8px;
    padding:6px 2px;
    color: var(--text-strong);
}
#toolbar select {
    background: #ffffff;
    color:var(--text-strong);
    border:1px solid var(--stroke);
    border-radius:10px;
    padding:7px 10px;
    font-size:12px;
    max-width:200px;
}
#toolbar button {
    min-height:34px;
    padding:7px 13px;
    border:1px solid var(--stroke);
    border-radius:var(--btn-radius);
    cursor:pointer;
    font-size:12px;
    font-weight:600;
    letter-spacing:0.15px;
    transition: background 0.14s ease, border-color 0.14s ease, color 0.14s ease;
    box-shadow: none;
    text-shadow: none;
}
#toolbar button:active { transform: none; }
#toolbar button:hover { background: var(--surface-2); }
#toolbar button:focus-visible {
    outline: 2px solid #84adff;
    outline-offset: 1px;
}
#toolbar .btn-nav,
#toolbar .btn-sel,
#toolbar .btn-era,
#toolbar .btn-undo,
#toolbar .btn-clear {
    background:#ffffff;
    color:var(--text-soft);
}
#toolbar .btn-merge {
    background:#ecfdf3;
    color:var(--success);
    border-color:#abefc6;
}
#toolbar .btn-split {
    background:#fff5f5;
    color:var(--danger);
    border-color:#fecdc9;
}
#toolbar .btn-delete {
    background:#fff5f5;
    color:var(--danger);
    border-color:#fecdc9;
}
#toolbar .btn-delete:hover {
    background:#fee4e2;
    border-color:#f97066;
}
#toolbar .btn-delete:disabled {
    opacity:0.4;
    cursor:not-allowed;
}

/* ── Delete Confirmation Modal ── */
#delete-modal {
    position:fixed;
    inset:0;
    z-index:1200;
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(16, 24, 40, 0.5);
    backdrop-filter: blur(1px);
}
#delete-panel {
    width:min(440px, 90vw);
    background:#ffffff;
    border:1px solid #fecdc9;
    border-radius:12px;
    padding:20px 22px;
    box-shadow: 0 14px 34px rgba(16, 24, 40, 0.2);
}
#delete-panel h3 {
    font-size:16px;
    color:#b42318;
    margin-bottom:10px;
}
#delete-panel p {
    font-size:13px;
    color:var(--text-soft);
    line-height:1.6;
    margin-bottom:16px;
}
#delete-panel .delete-actions {
    display:flex;
    justify-content:flex-end;
    gap:8px;
}
#delete-panel .delete-cancel {
    border:1px solid var(--stroke);
    background:#ffffff;
    color:var(--text-strong);
    border-radius:10px;
    padding:8px 14px;
    font-size:12px;
    font-weight:600;
    cursor:pointer;
}
#delete-panel .delete-cancel:hover { background:var(--surface-2); }
#delete-panel .delete-confirm {
    border:1px solid #f97066;
    background:#b42318;
    color:#ffffff;
    border-radius:10px;
    padding:8px 14px;
    font-size:12px;
    font-weight:600;
    cursor:pointer;
}
#delete-panel .delete-confirm:hover { background:#912018; }
#toolbar .btn-save {
    background:var(--primary);
    color:#ffffff;
    border-color:var(--primary);
}
#toolbar .btn-save:hover {
    background:var(--primary-hover);
    border-color:var(--primary-hover);
}
#toolbar .btn-clear {
    font-size:11px;
    padding:5px 10px;
}
#toolbar .btn-active {
    background:#f2f6ff !important;
    color:var(--primary) !important;
    border-color:#b2ccff !important;
}
#toolbar .sep { width:1px; height:26px; background:#e4e7ec; margin:0 1px; }
#toolbar .info { font-size:12px; color:var(--text-soft); }
#toolbar .badge {
    background:#ffffff;
    border:1px solid var(--stroke);
    border-radius: 999px;
    padding:5px 10px;
}
#toolbar .brush-badge {
    font-variant-numeric: tabular-nums;
    min-width: 82px;
    text-align: center;
}
#toolbar .hint {
    font-size:11px;
    color:#667085;
}
#toolbar .mode-ind {
    padding:5px 11px;
    border-radius:999px;
    font-weight:700;
    font-size:11px;
    letter-spacing:0.5px;
    border: 1px solid #d0d5dd;
}
#toolbar .mode-navigate { background:#eef4ff; color:#175cd3; }
#toolbar .mode-select { background:#fff7e6; color:#b54708; }
#toolbar .mode-eraser { background:#f4f7fb; color:#475467; }
#toolbar .btn-help {
    background:#ffffff;
    color:var(--text-soft);
}
#toolbar input[type='range'] { accent-color: #175cd3; }

/* ── Main Layout ── */
#main-area {
    position:fixed; top:58px; left:0; right:0; bottom:0;
    display:flex;
    overflow:hidden;
}
#canvas-area {
    flex:1; position:relative;
    background: #f3f6fb;
    min-width:0;
}
#legend {
    flex:0 0 var(--legend-width);
    width:var(--legend-width); min-width:220px; max-width:680px;
    background:var(--surface); border-left:1px solid var(--stroke);
    overflow-y:auto; padding:12px;
    box-shadow: none;
}
#legend-resizer {
    width:8px;
    flex:0 0 8px;
    cursor:col-resize;
    background:linear-gradient(to right, #eef2f7, #e2e8f0);
    border-left:1px solid #d7dee8;
    border-right:1px solid #d7dee8;
    position:relative;
    display:flex;
    align-items:center;
    justify-content:center;
}
#legend-resizer:hover {
    background:linear-gradient(to right, #dbe5f1, #cfd9e8);
}
#legend-resizer::before {
    content:'\2194';
    font-size:10px;
    color:#64748b;
    line-height:1;
    background:#f8fafc;
    border:1px solid #cbd5e1;
    border-radius:999px;
    width:18px;
    height:18px;
    display:flex;
    align-items:center;
    justify-content:center;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12);
    opacity:0.85;
    transition: transform 0.12s ease, opacity 0.12s ease;
}
#legend-resizer:hover::before {
    transform:scale(1.06);
    opacity:1;
}
body.legend-resizing,
body.legend-resizing * {
    cursor: col-resize !important;
}
#legend::-webkit-scrollbar { width:6px; }
#legend::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }

#legend h3 { font-size:15px; color:var(--text-strong); margin-bottom:8px; }
#legend .leg-ctrls { display:flex; gap:6px; margin-bottom:8px; }
#legend .leg-ctrls button {
    font-size:11px;
    font-weight:600;
    padding:5px 10px;
    border:1px solid #d4dce7;
    border-radius:8px;
    background:white;
    color:#334155;
    cursor:pointer;
}
#legend .leg-ctrls button:hover { background:#edf2f8; }

.leg-item {
    display:flex; align-items:center; padding:7px 8px; margin-bottom:4px;
    border-radius:10px; cursor:pointer; transition: all 0.15s;
    border: 1px solid #dde5ef; background:white;
}
.leg-item:hover { transform:none; background:#f4f8fd; }
.leg-item.dimmed { opacity:0.35; }
.leg-item.merge-selected {
    border-color:#b2ccff;
    background:#eef4ff;
    box-shadow: inset 0 0 0 1px #b2ccff;
}
.leg-item .dot { width:16px; height:16px; border-radius:50%; flex-shrink:0; border:2px solid rgba(0,0,0,0.1); }
.leg-item .lbl {
    flex:1;
    min-width:0;
    margin-left:8px;
    display:flex;
    align-items:baseline;
    gap:6px;
    font-size:13px;
    font-weight:600;
    color:var(--text-strong);
}
.leg-item .lbl-title {
    min-width:0;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
}
.leg-item .lbl-count {
    color:#9ca3af;
    font-size:11px;
    flex-shrink:0;
    white-space:nowrap;
}
.leg-item .merge-btn, .leg-item .vis-btn {
    background:none;
    border:1px solid transparent;
    cursor:pointer;
    padding:4px 7px;
    border-radius:8px;
    font-size:11px;
    font-weight:600;
    color:var(--text-soft);
    flex-shrink:0;
}
.leg-item .merge-btn:hover {
    color:var(--success);
    border-color:#abefc6;
    background:#ecfdf3;
}
.leg-item .vis-btn:hover {
    color:var(--primary);
    border-color:#b2ccff;
    background:#eef4ff;
}

/* ── Floating Elements ── */
#brush-cursor {
    position:fixed; pointer-events:none; z-index:1000;
    border: 2px solid #f59e0b; border-radius:50%;
    background: rgba(245,158,11,0.08);
    display:none;
}
#brush-cursor.eraser { border-color: #64748b; background: rgba(100,116,139,0.1); }

.tooltip-overlay {
    position:fixed; padding:6px 12px; background:rgba(0,0,0,0.8);
    color:white; border-radius:6px; font-size:12px; pointer-events:none;
    z-index:1100; display:none;
}

.loading-overlay {
    position:absolute; top:0; left:0; right:0; bottom:0;
    background: rgba(255,255,255,0.85); z-index:500;
    display:flex; align-items:center; justify-content:center;
    font-size:18px; font-weight:600; color:#666;
}

#help-modal {
    position:fixed;
    inset:0;
    z-index:1200;
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(16, 24, 40, 0.5);
    backdrop-filter: blur(1px);
}
#help-panel {
    width:min(720px, 92vw);
    max-height:80vh;
    overflow:auto;
    background:#ffffff;
    border:1px solid var(--stroke);
    border-radius:12px;
    padding:18px 20px;
    box-shadow: 0 14px 34px rgba(16, 24, 40, 0.2);
}
#help-panel h3 {
    font-size:17px;
    margin-bottom:8px;
    color:var(--text-strong);
}
#help-panel p {
    font-size:13px;
    color:var(--text-soft);
    margin-bottom:12px;
}
#help-panel .help-grid {
    display:grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap:10px;
}
#help-panel .help-card {
    border:1px solid #e4e7ec;
    border-radius:10px;
    padding:10px 12px;
    background:#fcfdff;
}
#help-panel .help-card h4 {
    font-size:12px;
    font-weight:700;
    margin-bottom:6px;
    color:var(--text-strong);
}
#help-panel .help-card ul {
    list-style:none;
}
#help-panel .help-card li {
    font-size:12px;
    color:var(--text-soft);
    line-height:1.55;
}
#help-panel .help-actions {
    margin-top:14px;
    display:flex;
    justify-content:flex-end;
}
#help-panel .help-close {
    border:1px solid var(--stroke);
    background:#ffffff;
    color:var(--text-strong);
    border-radius:10px;
    padding:8px 12px;
    font-size:12px;
    font-weight:600;
    cursor:pointer;
}

#status {
    position:fixed; bottom:12px; left:12px; z-index:200;
    background:rgba(0,0,0,0.85); padding:10px 16px; border-radius:8px;
    font-size:14px; color:#fff; display:none;
}
#boot-status {
    position:fixed; inset:0; z-index:2000;
    display:flex; align-items:center; justify-content:center;
    background:#f8fafc; color:#334155; font-size:16px; font-weight:600;
    padding:24px; text-align:center;
}
#boot-status.error {
    color:#991b1b; background:#fff7f7; white-space:pre-wrap;
}

@media (max-width: 1180px) {
    #toolbar {
        gap:6px;
        padding:8px;
        min-height:96px;
        align-items:flex-start;
        flex-wrap:wrap;
    }
    #main-area { top:96px; }
    #toolbar .title { margin-right:2px; }
}

@media (max-width: 860px) {
    #toolbar {
        min-height:128px;
    }
    #main-area {
        top:128px;
        flex-direction:column;
    }
    #canvas-area { min-height:58%; }
    #legend {
        width:100%;
        border-left:none;
        border-top:1px solid var(--stroke);
        box-shadow:none;
    }
    #legend-resizer { display:none; }
    #help-panel .help-grid {
        grid-template-columns: 1fr;
    }
}
</style>
</head>
<body>
<div id="boot-status">Loading frontend dependencies...</div>
<div id="app" v-cloak>

<!-- ═══════════════ TOOLBAR (plain flexbox like original) ═══════════════ -->
<div id="toolbar">
    <span class="title">Foram Cluster Editor</span>
    <select v-model.number="volIdx" @change="onVolumeChange(volIdx)">
        <option v-for="v in volumes" :key="v.idx" :value="v.idx">{{ v.name }}</option>
    </select>
    <div class="sep"></div>
    <button class="btn-nav" :class="{ 'btn-active': mode === 'navigate' }" @click="mode='navigate'">Navigate</button>
    <button class="btn-sel" :class="{ 'btn-active': mode === 'select' }" @click="mode='select'">Select</button>
    <button class="btn-era" :class="{ 'btn-active': mode === 'eraser' }" @click="mode='eraser'">Eraser</button>
    <button class="btn-undo" @click="undo" title="Undo (Ctrl+Z)">Undo</button>
    <span class="mode-ind" :class="'mode-' + mode">{{ mode.toUpperCase() }}</span>
    <div class="sep"></div>
    <button class="btn-clear" @click="clearSelection" title="Clear painted selection">Clear Selection</button>
    <div class="sep"></div>
    <button class="btn-merge" :class="{ 'btn-active': mergeClusterMode }" @click="doMerge" :title="mergeClusterMode ? 'Apply merge for selected chamber rows' : 'Enter chamber selection mode for cluster merge'">{{ mergeClusterMode ? 'Apply Merge' : 'Merge Clusters' }}</button>
    <button v-if="mergeClusterMode" class="btn-clear" @click="cancelMergeClusterSelection" title="Cancel chamber merge selection">Cancel Merge</button>
    <button class="btn-split" @click="doSplit">Split</button>
    <button class="btn-delete" @click="doDeleteSelected" :disabled="selectedCount === 0" :title="selectedCount === 0 ? 'Paint-select pores first to delete them' : 'Delete ' + selectedCount + ' selected pore(s) (with confirmation)'">Delete</button>
    <button class="btn-save" @click="doSave">Save</button>
    <button class="btn-help" @click="showHelp = true" title="Open user guide">Help</button>
    <div class="sep"></div>
    <span class="info badge brush-badge">Brush size: {{ brushSize }}</span>
    <span class="hint">Shift + Wheel</span>
</div>

<!-- ═══════════════ MAIN AREA ═══════════════ -->
<div id="main-area">
    <!-- 3D Canvas -->
    <div id="canvas-area" ref="threeContainer" 
         @mouseenter="mode !== 'navigate' ? (brushVisible = true) : null" 
         @mouseleave="brushVisible=false">
        <div class="loading-overlay" v-if="loading">Loading 3D Volume...</div>
    </div>

    <div id="legend-resizer" title="Drag to resize chamber panel"></div>

    <!-- Chambers Panel -->
    <div id="legend">
        <h3>Chambers <span style="font-size:11px;color:#9ca3af;font-weight:400;">(dbl-click=solo)</span></h3>
        <div class="leg-ctrls">
            <button @click="showAllChambers">Show All</button>
            <button @click="hideAllChambers">Hide All</button>
        </div>
        <div style="margin-bottom:12px; padding:0 4px;">
            <label style="font-size:13px; color:#4b5563; display:flex; align-items:center; gap:8px; cursor:pointer; margin-bottom:4px;">
                <input type="checkbox" v-model="showContext" @change="toggleContext(showContext)"> Show Chamber Context
            </label>
            <div style="display:flex; align-items:center; gap:8px; padding-left:24px;">
                <span style="font-size:11px; color:#6b7280; width:45px;">Opacity:</span>
                <input type="range" v-model.number="contextOpacity" min="0.05" max="1.0" step="0.05" style="flex:1;">
                <span style="font-size:11px; color:#6b7280; width:25px;">{{ Math.round(contextOpacity*100) }}%</span>
            </div>
        </div>

        <div v-if="mergeClusterMode" style="margin-bottom:10px; padding:8px 10px; border:1px solid #b2ccff; background:#eef4ff; border-radius:8px; font-size:12px; color:#1849a9;">
            Merge mode active: click chamber rows to pick clusters ({{ mergeClusterTargets.length }} selected), then click <b>Apply Merge</b>.
        </div>

        <div v-for="ch in chamberList" :key="ch.id"
             class="leg-item" :class="{ dimmed: ch.hidden, 'merge-selected': ch.mergeSelected }"
             @click="onChamberRowClick(ch.id)"
             @dblclick="!mergeClusterMode && soloChamber(ch.id)">
            <div class="dot" :style="{ background: ch.hex }"></div>
            <span class="lbl"><span class="lbl-title">Chamber {{ ch.displayIndex }}</span><span class="lbl-count">({{ ch.count }}p)</span></span>
            <button class="merge-btn" @click.stop="mergeSelectedInto(ch.id)" title="Move selected pores into this chamber">Merge Into</button>
            <button class="vis-btn" @click.stop="toggleChamber(ch.id)" :style="{ opacity: ch.hidden ? 0.3 : 0.7 }">
                {{ ch.hidden ? 'Show' : 'Hide' }}
            </button>
        </div>
    </div>
</div>

<div id="help-modal" v-if="showHelp" @click.self="showHelp = false">
    <div id="help-panel">
        <h3>Cluster Editor User Guide</h3>
        <p>Use this guide for the most common review and correction workflow. Press <b>Esc</b> to close this window.</p>
        <div class="help-grid">
            <div class="help-card">
                <h4>Navigation</h4>
                <ul>
                    <li>Left drag: rotate camera</li>
                    <li>Right drag: pan</li>
                    <li>Mouse wheel: zoom</li>
                    <li>In Select/Eraser, hold Shift + left drag to rotate</li>
                </ul>
            </div>
            <div class="help-card">
                <h4>Selection</h4>
                <ul>
                    <li>Select mode paints pores into the current selection</li>
                    <li>Eraser mode removes pores from selection</li>
                    <li>Shift + wheel adjusts brush radius</li>
                    <li>Clear resets the current selection</li>
                </ul>
            </div>
            <div class="help-card">
                <h4>Editing</h4>
                <ul>
                    <li>Merge Clusters: enter merge mode, pick chambers from the right panel, then Apply Merge</li>
                    <li>Split: creates a new chamber from selected pores</li>
                    <li>Merge Into (row button): moves current selection into that chamber</li>
                    <li>Use Show/Hide and double-click (solo) to isolate chambers</li>
                </ul>
            </div>
            <div class="help-card">
                <h4>Save and Undo</h4>
                <ul>
                    <li>Undo button or Ctrl+Z reverts last edit</li>
                    <li>Save writes corrected volume and mapping files</li>
                    <li>Saved files are placed under manual_review/{editor_state, corrected_volume, summary_npz, summary_tsv}</li>
                    <li>Status messages appear in the lower-left corner</li>
                </ul>
            </div>
        </div>
        <div class="help-actions">
            <button class="help-close" @click="showHelp = false">Close</button>
        </div>
    </div>
</div>


<div id="delete-modal" v-if="showDeleteConfirm" @click.self="cancelDelete">
    <div id="delete-panel">
        <h3>Delete Selected Pores?</h3>
        <p>
            You are about to permanently remove <b>{{ selectedCount }}</b> selected pore(s) from this volume.
            Deleted pores will be excluded from the saved output and will not appear in any chamber.<br><br>
            This action <b>can be undone</b> with Ctrl+Z before saving.
        </p>
        <div class="delete-actions">
            <button class="delete-cancel" @click="cancelDelete">Cancel</button>
            <button class="delete-confirm" @click="confirmDelete">Delete Pores</button>
        </div>
    </div>
</div>

</div> <!-- end #app -->

<div id="brush-cursor"></div>
<div id="hover-tooltip" class="tooltip-overlay"></div>
<div id="status"></div>

<script type="module">
const VUE_VERSION = "__VUE_VERSION__";
const THREE_VERSION = "__THREE_VERSION__";
const PROVIDERS = [
    {
        name: 'local',
        kind: 'local',
        vue: '/static/vue.esm-browser.prod.js',
        three: '/static/three.module.js',
        trackball: '/static/TrackballControls.js',
    },
    {
        name: 'jsdelivr',
        kind: 'remote',
        vue: `https://cdn.jsdelivr.net/npm/vue@${VUE_VERSION}/dist/vue.esm-browser.prod.js`,
        three: `https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/build/three.module.js`,
        trackball: `https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/examples/jsm/controls/TrackballControls.js`,
    },
    {
        name: 'unpkg',
        kind: 'remote',
        vue: `https://unpkg.com/vue@${VUE_VERSION}/dist/vue.esm-browser.prod.js`,
        three: `https://unpkg.com/three@${THREE_VERSION}/build/three.module.js`,
        trackball: `https://unpkg.com/three@${THREE_VERSION}/examples/jsm/controls/TrackballControls.js`,
    },
];


async function urlExists(url) {
    try {
        const response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
        return response.ok;
    } catch {
        return false;
    }
}

async function getAvailableProviders() {
    const available = [];
    for (const provider of PROVIDERS) {
        if (provider.kind !== 'local') {
            available.push(provider);
            continue;
        }
        const [hasVue, hasThree, hasTrackball] = await Promise.all([
            urlExists(provider.vue),
            urlExists(provider.three),
            urlExists(provider.trackball),
        ]);
        if (hasVue && hasThree && hasTrackball) {
            available.push(provider);
        } else {
            console.info('Skipping local frontend assets because one or more files are missing in /static/.');
        }
    }
    return available;
}

function setBootStatus(message, isError = false) {
    const el = document.getElementById('boot-status');
    if (!el) return;
    el.textContent = message;
    el.classList.toggle('error', isError);
}

async function importModuleFromUrl(url) {
    return await import(url);
}

async function importTrackballWithPatchedThree(provider) {
    const response = await fetch(provider.trackball, { mode: 'cors' });
    if (!response.ok) {
        throw new Error(`TrackballControls fetch failed (${response.status}) from ${provider.trackball}`);
    }
    let source = await response.text();
    const before = source;
    source = source.replace(/from\s*['"]three['"]/g, `from ${JSON.stringify(provider.three)}`);
    if (source === before && !provider.trackball.startsWith('/static/')) {
        throw new Error(`TrackballControls import patch did not match expected 'three' import in ${provider.trackball}`);
    }
    const blobUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
    try {
        const mod = await import(blobUrl);
        return mod.TrackballControls ?? mod.default;
    } finally {
        URL.revokeObjectURL(blobUrl);
    }
}

async function loadProviderBundle(provider) {
    const [vueMod, threeMod, TrackballControls] = await Promise.all([
        importModuleFromUrl(provider.vue),
        importModuleFromUrl(provider.three),
        importTrackballWithPatchedThree(provider),
    ]);
    return {
        provider: provider.name,
        Vue: vueMod.default ?? vueMod,
        THREE: threeMod,
        TrackballControls,
    };
}

async function loadFrontendModules() {
    const failures = [];
    const providers = await getAvailableProviders();
    for (const provider of providers) {
        try {
            setBootStatus(`Loading frontend dependencies from ${provider.name}...`);
            return await loadProviderBundle(provider);
        } catch (err) {
            const level = provider.kind === 'local' ? 'info' : 'warn';
            console[level](`Provider ${provider.name} failed`, err);
            failures.push(`${provider.name}: ${err?.message ?? err}`);
        }
    }
    const details = failures.length ? failures.join('\n') : 'No provider error details available.';
    throw new Error(`Unable to load Vue/Three.js/TrackballControls.\n\nAttempts:\n${details}`);
}

function startApp(Vue, THREE, TrackballControls) {
const VOLUMES = __DATA__;
const INITIAL_IDX = __INITIAL__;


// ── Palette ──
const PALETTE = [
    0xE6194B, 0x3CB44B, 0x4363D8, 0xF58231, 0x911EB4,
    0x42D4F4, 0xF032E6, 0xBFEF45, 0xFABED4, 0x469990,
    0xDCBEFF, 0x9A6324, 0x800000, 0xAAFFC3, 0x808000,
    0xFFD8B1, 0x000075, 0xA9A9A9, 0x000000, 0xE6BEFF,
];
const SEL_COLOR = new THREE.Color(0xFF0000);
const SHADOW_COLOR = new THREE.Color(0xFFCC00);

function chamberColor(cid) {
    if (cid === undefined || cid === null || isNaN(cid)) return new THREE.Color(0x999999);
    const idx = Math.max(0, Math.floor(cid));
    return new THREE.Color(PALETTE[idx % PALETTE.length]);
}
function chamberHex(cid) {
    if (cid === undefined || cid === null || isNaN(cid)) return '#999999';
    const idx = Math.max(0, Math.floor(cid));
    return '#' + PALETTE[idx % PALETTE.length].toString(16).padStart(6, '0');
}

// ── Three.js globals (initialized in initThree) ──
let renderer, scene, camera, controls;
let porePoints, contextPoints, poreGeom;
let centroids, voxelCoords, voxelOwners, chamberVoxels;
let labels = [];
let selected = new Set();
let hiddenChambers = new Set();
let historyStack = [];
let shiftHeld = false;
let legendResizing = false;
let painting = false;
let hoveringIndices = new Set(); // For paint preview "shadow"
let points2D = null; // Screen-space cache [x1, y1, z1, x2, y2, z2, ...]
let vm = null; // Vue instance reference
let brushCursor, tooltip, statusEl;
let projectionDirty = true;

function makeCircleTexture() {
    const sz = 64, c = document.createElement('canvas');
    c.width = sz; c.height = sz;
    const ctx = c.getContext('2d');
    ctx.beginPath(); ctx.arc(sz/2, sz/2, sz/2 - 2, 0, Math.PI*2);
    ctx.fillStyle = '#ffffff'; ctx.fill();
    return new THREE.CanvasTexture(c);
}

// ── Initializes Three.js AFTER Vue has mounted and the container exists ──
function initThree(container) {
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    container.insertBefore(renderer.domElement, container.firstChild);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xFFFFFF);

    camera = new THREE.PerspectiveCamera(50, 1, 1, 10000);

    controls = new TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 4.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.3;
    controls.staticMoving = false;
    controls.dynamicDampingFactor = 0.15;
    controls.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY ?? THREE.MOUSE.ZOOM,
        RIGHT: THREE.MOUSE.PAN,
    };

    controls.addEventListener('change', () => {
        projectionDirty = true;
        if (vm && (vm.mode === 'select' || vm.mode === 'eraser')) updateShadow();
    });

    scene.add(new THREE.AmbientLight(0xffffff, 1));

    const canvas = renderer.domElement;
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('mouseup', () => { painting = false; });
    brushCursor = document.getElementById('brush-cursor');
    tooltip = document.getElementById('hover-tooltip');
    statusEl = document.getElementById('status');

    document.addEventListener('mousemove', (e) => {
        if (vm && (vm.mode === 'select' || vm.mode === 'eraser')) {
            updateBrushCursor(e.clientX, e.clientY);
        }
    });

    window.addEventListener('keydown', (e) => {
        if (e.key === 'Shift') { shiftHeld = true; updateControlsMapping(); }
        if (e.key === 'Escape') {
            if (vm && vm.showHelp) vm.showHelp = false;
            else if (vm && vm.mergeClusterMode) vm.cancelMergeClusterSelection();
            else if (vm) vm.mode = 'navigate';
        }
        if (e.ctrlKey && e.key === 'z') { e.preventDefault(); if (vm) vm.undo(); }
    });
    window.addEventListener('keyup', (e) => {
        if (e.key === 'Shift') { shiftHeld = false; updateControlsMapping(); }
    });

    function onResize() {
        const w = container.clientWidth, h = container.clientHeight;
        if (w === 0 || h === 0) return;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        projectionDirty = true;
        ensureProjectionCache();
        if (controls.handleResize) controls.handleResize();
    }
    window.addEventListener('resize', onResize);
    onResize();

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    updateControlsMapping();
}

const circleTexture = makeCircleTexture();

function buildScene(cens, chamberV) {
    if (!scene) return;
    while (scene.children.length > 0) scene.remove(scene.children[0]);

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(100, 200, 100);
    scene.add(sun);

    if (chamberV && chamberV.length > 0) {
        const cGeom = new THREE.BufferGeometry();
        const cPos = new Float32Array(chamberV.length * 3);
        for (let i = 0; i < chamberV.length; i++) {
            cPos[i*3] = chamberV[i][2]; cPos[i*3+1] = chamberV[i][1]; cPos[i*3+2] = chamberV[i][0];
        }
        cGeom.setAttribute('position', new THREE.BufferAttribute(cPos, 3));
        const cMat = new THREE.PointsMaterial({
            color: 0x94a3b8, size: 2.0, transparent: true, opacity: vm ? vm.contextOpacity : 0.20, depthWrite: false
        });
        contextPoints = new THREE.Points(cGeom, cMat);
        contextPoints.visible = vm ? vm.showContext : false;
        scene.add(contextPoints);
    } else {
        contextPoints = null;
    }

    const useVoxels = (voxelCoords && voxelCoords.length > 0);
    const nPts = useVoxels ? voxelCoords.length : (cens ? cens.length : 0);
    const posSource = useVoxels ? voxelCoords : cens;

    if (nPts === 0) return;

    poreGeom = new THREE.BufferGeometry();
    const pp = new Float32Array(nPts * 3);
    const pc = new Float32Array(nPts * 3);
    for (let i = 0; i < nPts; i++) {
        pp[i*3] = posSource[i][2]; pp[i*3+1] = posSource[i][1]; pp[i*3+2] = posSource[i][0];
    }
    poreGeom.setAttribute('position', new THREE.BufferAttribute(pp, 3));
    poreGeom.setAttribute('color', new THREE.BufferAttribute(pc, 3));

    const poreMat = new THREE.PointsMaterial({
        size: useVoxels ? 2.5 : 8,
        vertexColors: true, sizeAttenuation: true,
        map: circleTexture, alphaTest: 0.5, transparent: true
    });
    porePoints = new THREE.Points(poreGeom, poreMat);
    scene.add(porePoints);

    points2D = new Float32Array(nPts * 3);
    projectionDirty = true;
    ensureProjectionCache();
    updateColors();

    poreGeom.computeBoundingSphere();
    const bc = poreGeom.boundingSphere.center;
    const br = poreGeom.boundingSphere.radius;
    camera.position.set(bc.x + br*0.5, bc.y + br*0.5, bc.z + br * 2.25);
    controls.target.copy(bc);
    controls.update();
    projectionDirty = true;
    ensureProjectionCache();
}

function ensureProjectionCache() {
    if (!projectionDirty) return;
    updateProjectionCache();
    projectionDirty = false;
}

function updateProjectionCache() {
    if (!poreGeom || !points2D || !renderer || !camera) return;
    const pos = poreGeom.attributes.position.array;
    const n = pos.length / 3;
    const canvas = renderer.domElement;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const projScreenMatrix = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const v = new THREE.Vector3();

    for (let i = 0; i < n; i++) {
        v.set(pos[i*3], pos[i*3+1], pos[i*3+2]);
        v.applyMatrix4(projScreenMatrix);
        points2D[i*3] = (v.x * 0.5 + 0.5) * w;
        points2D[i*3+1] = (-v.y * 0.5 + 0.5) * h;
        points2D[i*3+2] = v.z;
    }
}

function updateColors() {
    if (!poreGeom || !poreGeom.attributes.color) return;
    const cols = poreGeom.attributes.color.array;
    const useVoxels = (voxelCoords && voxelCoords.length > 0);
    const nPts = useVoxels ? voxelCoords.length : (centroids ? centroids.length : 0);
    for (let i = 0; i < nPts; i++) {
        const poreIdx = useVoxels ? voxelOwners[i] : i;
        const cid = labels[poreIdx];
        let c;
        if (cid === 0) c = new THREE.Color(1.0, 1.0, 1.0); // deleted → invisible on white bg
        else if (selected.has(poreIdx)) c = SEL_COLOR;
        else if (hoveringIndices.has(i)) c = SHADOW_COLOR;
        else if (hiddenChambers.has(cid)) c = new THREE.Color(0.95, 0.95, 0.95);
        else c = chamberColor(cid);
        cols[i*3] = c.r; cols[i*3+1] = c.g; cols[i*3+2] = c.b;
    }
    poreGeom.attributes.color.needsUpdate = true;
}

function updateControlsMapping() {
    if (!controls || !vm || !renderer) return;
    const zoomButton = THREE.MOUSE.DOLLY ?? THREE.MOUSE.ZOOM;
    const isPaintMode = vm.mode === 'select' || vm.mode === 'eraser';

    // Prevent wheel-zoom collision when Shift+scroll is used for brush resizing.
    controls.noZoom = isPaintMode && shiftHeld;
    if (isPaintMode) {
        controls.mouseButtons.LEFT = shiftHeld ? THREE.MOUSE.ROTATE : null;
        controls.mouseButtons.MIDDLE = zoomButton;
        controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
        renderer.domElement.style.cursor = 'crosshair';
    } else {
        controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
        controls.mouseButtons.MIDDLE = zoomButton;
        controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
        renderer.domElement.style.cursor = 'grab';
    }
}

function paintAt(mx, my, erasing) {
    if (!vm || !points2D) return;
    ensureProjectionCache();
    const bR2 = vm.brushSize * vm.brushSize;
    let changed = false;
    const useVoxels = (voxelCoords && voxelCoords.length > 0);
    const nPts = useVoxels ? voxelCoords.length : (centroids ? centroids.length : 0);
    for (let i = 0; i < nPts; i++) {
        const pz = points2D[i*3+2]; if (pz < 0 || pz > 1) continue;
        const dx = points2D[i*3] - mx, dy = points2D[i*3+1] - my;
        if (dx*dx + dy*dy < bR2) {
            const poreIdx = useVoxels ? voxelOwners[i] : i;
            if (labels[poreIdx] === 0) continue; // skip deleted pores
            if (hiddenChambers.has(labels[poreIdx])) continue;
            if (erasing) { if (selected.has(poreIdx)) { selected.delete(poreIdx); changed = true; } }
            else { if (!selected.has(poreIdx)) { selected.add(poreIdx); changed = true; } }
        }
    }
    if (changed) { updateColors(); vm.selectedCount = selected.size; }
}

function updateShadow() {
    if (!vm || !points2D || !renderer || !porePoints) return;
    ensureProjectionCache();
    const mx = lastMX, my = lastMY;
    const bR2 = vm.brushSize * vm.brushSize;
    const n = points2D.length / 3;
    hoveringIndices.clear();
    const useVoxels = (voxelCoords && voxelCoords.length > 0);
    for (let i = 0; i < n; i++) {
        const pz = points2D[i*3+2]; if (pz < 0 || pz > 1) continue;
        const dx = points2D[i*3] - mx, dy = points2D[i*3+1] - my;
        if (dx*dx + dy*dy < bR2) {
            const poreIdx = useVoxels ? voxelOwners[i] : i;
            const cid = labels[poreIdx];
            if (cid !== undefined && cid !== 0 && !hiddenChambers.has(cid)) hoveringIndices.add(i);
        }
    }
    updateColors();
}

let lastMX = 0, lastMY = 0;
let lastClientX = 0, lastClientY = 0;
let hasPointerSample = false;

function updateBrushCursor(x, y) {
    if (!vm || !brushCursor) return;
    if (!renderer || !renderer.domElement) {
        brushCursor.style.display = 'none';
        return;
    }
    const rect = renderer.domElement.getBoundingClientRect();
    const inside = x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
    if (!inside) {
        brushCursor.style.display = 'none';
        return;
    }
    const r = vm.brushSize;
    brushCursor.style.width = r*2+'px'; brushCursor.style.height = r*2+'px';
    brushCursor.style.left = (x-r)+'px'; brushCursor.style.top = (y-r)+'px';
    brushCursor.style.display = vm.brushVisible ? 'block' : 'none';
}

function onMouseDown(e) {
    if (!vm) return;
    if ((vm.mode === 'select' || vm.mode === 'eraser') && e.button === 0 && !e.shiftKey) {
        painting = true;
        const rect = renderer.domElement.getBoundingClientRect();
        paintAt(e.clientX - rect.left, e.clientY - rect.top, vm.mode === 'eraser');
    }
}

function onMouseMove(e) {
    if (!vm) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    lastMX = mx; lastMY = my;
    lastClientX = e.clientX; lastClientY = e.clientY;
    hasPointerSample = true;

    if (vm.mode === 'select' || vm.mode === 'eraser') {
        updateBrushCursor(e.clientX, e.clientY);
        updateShadow();
        if (painting && !e.shiftKey) paintAt(mx, my, vm.mode === 'eraser');
    }
    if (porePoints && renderer && renderer.domElement) {
        const mouseDom = renderer.domElement;
        if (mouseDom.clientWidth > 0 && mouseDom.clientHeight > 0) {
            const mouseVec = new THREE.Vector2(
                (mx / mouseDom.clientWidth) * 2 - 1,
                -(my / mouseDom.clientHeight) * 2 + 1
            );
            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(mouseVec, camera);
            raycaster.params.Points.threshold = 3;
            const intersects = raycaster.intersectObject(porePoints);
            if (intersects.length > 0) {
                const idx = intersects[0].index;
                const useVoxels = (voxelCoords && voxelCoords.length > 0);
                const poreIdx = useVoxels ? voxelOwners[idx] : idx;
                const cid = labels[poreIdx];
                if (cid !== undefined && cid !== 0) {
                    const displayIndex = vm ? vm.getDisplayIndex(cid) : (cid - 1);
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.clientX + 15) + 'px';
                    tooltip.style.top = (e.clientY + 15) + 'px';
                    tooltip.innerHTML = `<b>Chamber ${displayIndex}</b><br><small>Pore ${poreIdx}</small>`;
                    return;
                }
            }
        }
    }
    if (tooltip) tooltip.style.display = 'none';
}

function onWheel(e) {
    if (!vm) return;
    const isPaintMode = vm.mode === 'select' || vm.mode === 'eraser';
    if (!isPaintMode) return;
    if (!e.shiftKey) return;

    e.preventDefault();
    e.stopPropagation();
    const step = e.deltaMode === 1 ? 2 : 4;
    const next = vm.brushSize + (e.deltaY > 0 ? -step : step);
    vm.brushSize = Math.max(5, Math.min(250, next));

    // Apply size change immediately without waiting for mousemove/key release.
    if (hasPointerSample) {
        const rect = renderer.domElement.getBoundingClientRect();
        lastMX = lastClientX - rect.left;
        lastMY = lastClientY - rect.top;
        updateBrushCursor(lastClientX, lastClientY);
        updateShadow();
    }
}

function initLegendResizer() {
    const resizer = document.getElementById('legend-resizer');
    const legend = document.getElementById('legend');
    if (!resizer) return;
    if (!legend) return;

    let startX = 0;
    let startWidth = 300;

    const applyLegendWidth = (px) => {
        const next = Math.max(220, Math.min(680, px));
        document.documentElement.style.setProperty('--legend-width', `${next}px`);
        legend.style.width = `${next}px`;
        legend.style.flexBasis = `${next}px`;
    };

    const stopResizing = () => {
        if (!legendResizing) return;
        legendResizing = false;
        document.body.classList.remove('legend-resizing');
        document.body.style.userSelect = '';
    };

    resizer.addEventListener('pointerdown', (e) => {
        if (window.innerWidth <= 860) return;
        if (e.button !== 0) return;
        legendResizing = true;
        startX = e.clientX;
        startWidth = legend.getBoundingClientRect().width;
        if (resizer.setPointerCapture) {
            resizer.setPointerCapture(e.pointerId);
        }
        document.body.classList.add('legend-resizing');
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    window.addEventListener('pointermove', (e) => {
        if (!legendResizing) return;
        const dx = e.clientX - startX;
        applyLegendWidth(startWidth - dx);
        window.dispatchEvent(new Event('resize'));
    });

    window.addEventListener('pointerup', (e) => {
        if (!legendResizing) return;
        if (resizer.releasePointerCapture) {
            try { resizer.releasePointerCapture(e.pointerId); } catch (_) {}
        }
        stopResizing();
    });

    window.addEventListener('pointercancel', () => {
        stopResizing();
    });

    resizer.addEventListener('lostpointercapture', () => {
        stopResizing();
    });

    window.addEventListener('blur', () => {
        stopResizing();
    });
}

function reindex() {
    // label 0 = deleted pore; exclude from remapping so deleted pores stay as 0
    // Sort by descending pore count so label 2 always has the most pores,
    // label 3 the second most, etc. — consistent with the reindex_clusters pipeline.
    const counts = {};
    for (let i = 0; i < labels.length; i++) {
        const v = labels[i];
        if (v >= 2) counts[v] = (counts[v] || 0) + 1;
    }
    const unique = Object.keys(counts)
        .map(Number)
        .sort((a, b) => counts[b] - counts[a] || a - b);  // desc count, asc id for ties
    const map = {};
    unique.forEach((old, i) => { map[old] = i + 2; });
    for (let i = 0; i < labels.length; i++) {
        if (labels[i] >= 2) labels[i] = map[labels[i]];
    }
    const nextHidden = new Set();
    hiddenChambers.forEach(oldId => { const nextId = map[oldId]; if (nextId) nextHidden.add(nextId); });
    hiddenChambers = nextHidden;
}

function pushHistory() {
    historyStack.push([...labels]);
    if (historyStack.length > 50) historyStack.shift();
}

let statusTimer = null;

const app = Vue.createApp({
    data() {
        return {
            volumes: VOLUMES,
            volIdx: INITIAL_IDX,
            mode: 'navigate',
            brushSize: 20,
            selectedCount: 0,
            loading: false,
            showContext: false,
            contextOpacity: 0.2,
            brushVisible: false,
            showHelp: false,
            chamberList: [],
            chamberIndexMap: {},
            mergeClusterMode: false,
            mergeClusterTargets: [],
            showDeleteConfirm: false,
        };
    },

    watch: {
        mode(newMode) {
            if (newMode === 'navigate') this.clearSelection();
            if (brushCursor) {
                this.brushVisible = (newMode === 'select' || newMode === 'eraser');
                brushCursor.classList.toggle('eraser', newMode === 'eraser');
            }
            updateControlsMapping();
        },
        brushVisible(val) {
            if (brushCursor) brushCursor.style.display = val ? 'block' : 'none';
        },
        contextOpacity(newVal) {
            if (contextPoints) {
                contextPoints.material.opacity = newVal;
                contextPoints.material.needsUpdate = true;
            }
        }
    },
    methods: {
        async onVolumeChange(idx) {
            this.loading = true;
            try {
                const volumeMeta = this.volumes[idx];
                const resp = await fetch('/api/volume/' + idx);
                if (!resp.ok) throw new Error('API error: ' + resp.status);
                const vol = await resp.json();

                centroids = vol.centroids;
                labels = [...vol.labels];
                voxelCoords = vol.pore_voxels;
                voxelOwners = vol.pore_voxels_owner;
                chamberVoxels = vol.chamber_voxels;
                hiddenChambers.clear();
                selected.clear();
                hoveringIndices.clear();
                historyStack = [];
                this.selectedCount = 0;
                this.mergeClusterMode = false;
                this.mergeClusterTargets = [];

                buildScene(vol.centroids, chamberVoxels);
                this.rebuildChamberList();
                if (volumeMeta && volumeMeta.corrected) {
                    this.showStatus(`Loaded corrected manual_review state for: ${volumeMeta.name}`, 12000);
                }
            } catch (e) {
                console.error('Failed to load volume:', e);
                this.showStatus('Error loading volume');
            } finally {
                this.loading = false;
            }
        },
        rebuildChamberList() {
            if (!labels || labels.length === 0) { this.chamberList = []; return; }
            const counts = {};
            for (let i = 0; i < labels.length; i++) {
                const cid = labels[i];
                if (cid === undefined || cid === null || isNaN(cid)) continue;
                if (cid < 2) continue;
                counts[cid] = (counts[cid] || 0) + 1;
            }
            const ordered = Object.keys(counts)
                .map(Number)
                .sort((a, b) => counts[b] - counts[a] || a - b);
            const limitedIds = ordered.slice(0, 500);
            const idxMap = {};
            limitedIds.forEach((cid, index) => { idxMap[cid] = index + 1; });
            this.chamberIndexMap = idxMap;
            this.chamberList = limitedIds.map(cid => ({
                id: cid,
                displayIndex: idxMap[cid],
                count: counts[cid],
                hex: chamberHex(cid),
                hidden: hiddenChambers.has(cid),
                mergeSelected: this.mergeClusterTargets.includes(cid),
            }));
        },
        onChamberRowClick(cid) {
            if (!this.mergeClusterMode) return;
            const idx = this.mergeClusterTargets.indexOf(cid);
            if (idx >= 0) this.mergeClusterTargets.splice(idx, 1);
            else this.mergeClusterTargets.push(cid);
            this.rebuildChamberList();
        },
        cancelMergeClusterSelection() {
            if (!this.mergeClusterMode) return;
            this.mergeClusterMode = false;
            this.mergeClusterTargets = [];
            this.rebuildChamberList();
            this.showStatus('Merge Clusters cancelled.');
        },
        getDisplayIndex(cid) {
            return this.chamberIndexMap[cid] ?? (cid - 1);
        },
        toggleChamber(cid) {
            if (hiddenChambers.has(cid)) hiddenChambers.delete(cid);
            else hiddenChambers.add(cid);
            updateColors(); this.rebuildChamberList();
        },
        showAllChambers() { hiddenChambers.clear(); updateColors(); this.rebuildChamberList(); },
        hideAllChambers() {
            [...new Set(labels)].forEach(cid => { if (cid >= 2) hiddenChambers.add(cid); });
            updateColors(); this.rebuildChamberList();
        },
        soloChamber(cid) {
            const ids = [...new Set(labels)].filter(c => c >= 2).sort((a, b) => a - b);
            const visible = ids.filter(c => !hiddenChambers.has(c));
            if (visible.length === 1 && visible[0] === cid) {
                hiddenChambers.clear();
            } else {
                hiddenChambers.clear();
                ids.forEach(c => { if (c !== cid) hiddenChambers.add(c); });
            }
            updateColors(); this.rebuildChamberList();
        },
        toggleContext(val) { if (contextPoints) contextPoints.visible = val; },
        clearSelection() { selected.clear(); hoveringIndices.clear(); updateColors(); this.selectedCount = 0; },
        doMerge() {
            if (!this.mergeClusterMode) {
                this.mergeClusterMode = true;
                this.mergeClusterTargets = [];
                this.rebuildChamberList();
                this.showStatus('Merge Clusters mode: pick chamber rows, then click Apply Merge.');
                return;
            }

            const chambers = new Set(this.mergeClusterTargets);
            if (chambers.size < 2) {
                this.showStatus('Apply Merge needs at least 2 selected chamber rows.');
                return;
            }

            pushHistory();
            const target = Math.min(...chambers);
            for (let i = 0; i < labels.length; i++) {
                if (chambers.has(labels[i])) labels[i] = target;
            }

            reindex();
            selected.clear();
            hoveringIndices.clear();
            this.selectedCount = 0;
            this.mergeClusterMode = false;
            this.mergeClusterTargets = [];
            updateColors(); this.rebuildChamberList();
            this.showStatus('Merge Clusters: merged selected chamber rows.');
        },
        mergeSelectedInto(targetCid) {
            if (selected.size === 0) { this.showStatus('Merge Into requires a current painted selection.'); return; }
            pushHistory();
            let moved = 0;
            selected.forEach(pid => {
                if (labels[pid] !== targetCid) moved += 1;
                labels[pid] = targetCid;
            });
            reindex(); selected.clear(); this.selectedCount = 0;
            updateColors(); this.rebuildChamberList();
            this.showStatus(`Merge Into: ${moved} newly moved pores into Chamber ${this.getDisplayIndex(targetCid)}`);
        },
        doSplit() {
            if (selected.size === 0) { this.showStatus('Paint-select pores first.'); return; }
            pushHistory();
            const n = selected.size;
            const newIdx = Math.max(...labels) + 1;
            selected.forEach(pid => { labels[pid] = newIdx; });
            reindex(); selected.clear(); this.selectedCount = 0;
            updateColors(); this.rebuildChamberList();
            this.showStatus(`Split ${n} pores into new Chamber ${this.getDisplayIndex(newIdx)}`);
        },
        async doSave() {
            const vol = this.volumes[this.volIdx];
            try {
                const resp = await fetch('/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file: vol.file, labels: labels }),
                });
                if (!resp.ok) throw new Error(`Save failed: ${resp.status}`);
                const r = await resp.json();
                if (r.saved_state) vol.file = r.saved_state;
                vol.corrected = true;
                this.showStatus('Saved to manual_review: ' + (r.saved_state || r.saved_volume));
            } catch (err) {
                console.error(err);
                this.showStatus('Save failed');
            }
        },
        doDeleteSelected() {
            if (selected.size === 0) { this.showStatus('Paint-select pores first to delete them.'); return; }
            this.showDeleteConfirm = true;
        },
        cancelDelete() {
            this.showDeleteConfirm = false;
        },
        confirmDelete() {
            const n = selected.size;
            pushHistory();
            selected.forEach(pid => { labels[pid] = 0; });
            reindex();
            selected.clear();
            hoveringIndices.clear();
            this.selectedCount = 0;
            this.showDeleteConfirm = false;
            updateColors();
            this.rebuildChamberList();
            this.showStatus(`Deleted ${n} pore(s). They will be excluded from the saved volume.`);
        },
        undo() {
            if (historyStack.length === 0) { this.showStatus('Nothing to undo.'); return; }
            labels = historyStack.pop(); selected.clear(); hoveringIndices.clear(); this.selectedCount = 0;
            updateColors(); this.rebuildChamberList();
            this.showStatus('Undo successful.');
        },
        showStatus(msg, duration = 6000) {
            if (!statusEl) return;
            statusEl.textContent = msg;
            statusEl.style.display = 'block';
            if (statusTimer) clearTimeout(statusTimer);
            statusTimer = setTimeout(() => { if (statusEl) statusEl.style.display = 'none'; }, duration);
        }
    },
    mounted() {
        vm = this;
        this.$nextTick(() => {
            initLegendResizer();
            const container = this.$refs.threeContainer;
            if (container) {
                initThree(container);
                this.onVolumeChange(this.volIdx);
            }
        });
    }
});
app.mount('#app');
}

try {
    const { provider, Vue, THREE, TrackballControls } = await loadFrontendModules();
    setBootStatus(`Loaded frontend dependencies from ${provider}.`);
    startApp(Vue, THREE, TrackballControls);
    const el = document.getElementById('boot-status');
    if (el) el.style.display = 'none';
} catch (err) {
    console.error(err);
    setBootStatus(`Frontend failed to initialize.

${err.message}`, true);
}
</script>
</body>
</html>'''


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _listener_uids_on_port(port):
    uids = set()
    port_hex = f":{port:04X}"
    for path in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                next(f, None)
                for line in f:
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    local_addr = parts[1]
                    state = parts[3]
                    uid = parts[7]
                    if state == '0A' and local_addr.endswith(port_hex):
                        uids.add(int(uid))
        except OSError:
            continue
    return sorted(uids)


def _uid_to_name(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def find_open_port(host, start_port, max_tries=50):
    port = start_port
    for _ in range(max_tries):
        try:
            probe = ThreadedHTTPServer((host, port), EditorHandler)
            return probe, port
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                port += 1
                continue
            raise
    raise OSError(errno.EADDRINUSE, f"No free port found in range {start_port}-{port}")

def main():
    parser = argparse.ArgumentParser(description='3D Cluster Editor (Vue Edition)')
    parser.add_argument('folder', help='Folder containing .npz cluster state files')
    parser.add_argument('--host', default='0.0.0.0', help='Host/interface to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument(
        '--auto-port',
        action='store_true',
        help='If requested port is busy, try subsequent ports until one is available',
    )
    args = parser.parse_args()
    folder = os.path.abspath(args.folder)

    if not os.path.isdir(folder):
        print(f'Error: {args.folder} is not a directory')
        sys.exit(1)

    all_npz = sorted(glob.glob(os.path.join(folder, '*.npz')))
    npz_files = [f for f in all_npz if is_editor_state_npz(f)]

    if not npz_files:
        print(f'No valid cluster state .npz files found in {folder}')
        sys.exit(1)

    npz_files, corrected_names = prefer_manual_review_states(npz_files, folder)

    print(f'Found {len(npz_files)} volumes:')
    for f in npz_files:
        print(f'  - {os.path.basename(f)}')
    if corrected_names:
        preview = ', '.join(corrected_names[:10])
        suffix = '' if len(corrected_names) <= 10 else f', ... ({len(corrected_names)} total)'
        print(f'Using corrected manual_review state for: {preview}{suffix}')

    STATE['static_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    build_html(npz_files, corrected_names=corrected_names)

    try:
        if args.auto_port:
            server, chosen_port = find_open_port(args.host, args.port)
        else:
            server = ThreadedHTTPServer((args.host, args.port), EditorHandler)
            chosen_port = args.port
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(f"Error: Port {args.port} is already in use on host {args.host}.")
            owners = _listener_uids_on_port(args.port)
            if owners:
                owner_names = ', '.join(_uid_to_name(uid) for uid in owners)
                print(f"Detected listener owner UID(s): {owner_names}")
            print('Tip: retry with --auto-port or choose another port, e.g. --port 5006')
            sys.exit(1)
        raise

    if args.auto_port and chosen_port != args.port:
        print(f"Requested port {args.port} was busy; using port {chosen_port} instead.")

    print(f'\nStarting Vue Editor on http://localhost:{chosen_port}')
    print('Press Ctrl+C to stop.\n')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nEditor stopped.')
        server.server_close()


if __name__ == '__main__':
    main()
