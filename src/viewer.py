"""Local reconstruction gallery and browser viewer."""
import argparse
import json
import shutil
from pathlib import Path
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

OUTPUT_FILES = ('sparse.ply', 'point_cloud.ply', 'mesh.ply', 'voxels.npz',
                'metrics.json', 'comparison.png', 'colmap.log')


def read_json(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def create_app(output_dir, gallery_root=None):
    app = Flask(__name__)
    default_output = Path(output_dir).resolve()
    root = Path(gallery_root).resolve() if gallery_root else default_output.parent
    static_dir = Path(__file__).resolve().parent.parent / 'web'

    def projects():
        found = {}
        candidates = []
        if gallery_root:
            candidates.append(Path(gallery_root).resolve())
        else:
            outputs_folder = root / 'outputs' if root.name != 'outputs' else root
            if outputs_folder.is_dir():
                candidates.append(outputs_folder.resolve())
            if default_output.is_dir() and default_output not in candidates:
                candidates.append(default_output)
            candidates.append(root)

        for base in candidates:
            if base.is_dir():
                for folder in base.iterdir():
                    if not folder.is_dir() or folder.name.startswith('.'):
                        continue
                    if any((folder/name).is_file() for name in (*OUTPUT_FILES, 'project.json')):
                        if folder.name not in found:
                            found[folder.name] = folder.resolve()
        return found

    def selected_output():
        key = request.args.get('project')
        all_proj = projects()
        if key is None:
            if all_proj:
                return next(iter(all_proj.values()))
            return default_output
        folder = all_proj.get(key)
        if folder is None:
            abort(404, description='Reconstruction not found')
        return folder

    def description(folder):
        files = {name: (folder/name).is_file() for name in OUTPUT_FILES}
        metrics = read_json(folder/'metrics.json')
        metadata = read_json(folder/'project.json')
        name = metadata.get('name') or folder.name.removeprefix('outputs_').replace('_', ' ').title()
        state = ('Complete' if files['metrics.json'] and files['mesh.ply'] and files['voxels.npz']
                 else 'Dense model available' if files['point_cloud.ply']
                 else 'Sparse preview available' if files['sparse.ply'] else 'Preparing reconstruction')
        preview = 'preview.jpg' if (folder/'preview.jpg').is_file() else 'comparison.png' if files['comparison.png'] else None
        return {'id': folder.name, 'name': name, 'files': files, 'metrics': metrics,
                'state': state, 'preview': preview,
                'preview_kind': 'Source photo' if preview == 'preview.jpg' else 'Reconstruction comparison',
                'input_images': metadata.get('input_images', metrics.get('input_images')),
                'updated': max((p.stat().st_mtime for p in folder.iterdir() if p.is_file()), default=0)}

    @app.get('/')
    def index():
        return send_from_directory(static_dir, 'gallery.html')

    @app.get('/view')
    def view():
        selected_output()
        return send_from_directory(static_dir, 'index.html')

    @app.get('/api/projects')
    def list_projects():
        return jsonify(sorted((description(p) for p in projects().values()),
                              key=lambda p: p['updated'], reverse=True))

    @app.get('/api/status')
    def status():
        folder = selected_output()
        if not folder.exists():
            return jsonify({'files': {name: False for name in OUTPUT_FILES}, 'metrics': {}})
        return jsonify(description(folder))

    @app.delete('/api/projects/<project_id>')
    def delete_project(project_id):
        # JSON and same-origin checks prevent cross-site deletion requests.
        if request.headers.get('Origin') and request.headers['Origin'] != request.host_url.rstrip('/'):
            return jsonify(error='Cross-origin deletion is not allowed.'), 403
        data = request.get_json(silent=True) or {}
        if data.get('confirm') != project_id:
            return jsonify(error='Confirm the selected reconstruction before deleting.'), 400
        folder = projects().get(project_id)
        if folder is None:
            return jsonify(error='Reconstruction not found.'), 404
        if folder.is_symlink() or folder not in projects().values():
            return jsonify(error='Linked folders cannot be deleted.'), 400
        source = read_json(folder/'project.json').get('source')
        if source:
            source = Path(source).resolve()
            if source == folder or folder in source.parents:
                return jsonify(error='This folder contains source photos and cannot be deleted.'), 409
        try:
            shutil.rmtree(folder)
        except OSError:
            return jsonify(error='Could not fully delete this reconstruction. Close programs using its files and retry.'), 409
        return jsonify(deleted=project_id)

    @app.get('/api/file/<name>')
    def file(name):
        if name not in ('sparse.ply', 'point_cloud.ply', 'mesh.ply', 'comparison.png', 'preview.jpg'):
            return jsonify({'error': 'Unsupported file'}), 404
        folder = selected_output()
        path = (folder/name).resolve()
        if path.parent != folder or not path.is_file():
            return jsonify({'error': 'Reconstruction output not found'}), 404
        return send_file(path)

    @app.get('/api/voxels')
    def voxels():
        folder = selected_output()
        path = (folder/'voxels.npz').resolve()
        if path.parent != folder or not path.is_file():
            return jsonify({'error': 'Run full reconstruction first'}), 404
        import numpy as np
        with np.load(path) as data:
            indices = data['indices']
            step = max(1, (len(indices)+79999)//80000)
            return jsonify({'indices': indices[::step].tolist(), 'origin': data['origin'].tolist(),
                            'size': float(data['voxel_size']), 'total': len(indices),
                            'displayed': len(indices[::step])})
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--gallery-root', help='Folder containing saved reconstruction folders')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print(f'Open http://{args.host}:{args.port}')
    create_app(args.output, args.gallery_root).run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
