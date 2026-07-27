import re
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
target_file = os.path.join(project_root, 'src', 'post_processing', 'cluster_editor.py')

with open(target_file, 'r') as f:
    content = f.read()

# Replace OrbitControls import with TrackballControls
content = content.replace("import { OrbitControls } from 'three/addons/controls/OrbitControls.js';", "import { TrackballControls } from 'three/addons/controls/TrackballControls.js';")

# Setup TrackballControls
old_controls = """// Orbit controls — always enabled for RIGHT mouse button
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.minPolarAngle = 0;          // Full 360°
controls.maxPolarAngle = Math.PI;    // Full 360°
controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
};"""

new_controls = """// Trackball controls allow for true 360 free-form rotation without polar locks
const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 4.0;
controls.zoomSpeed = 1.2;
controls.panSpeed = 0.8;
controls.staticMoving = false;
controls.dynamicDampingFactor = 0.15;
// Default mouse buttons for Trackball
controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.ZOOM,
    RIGHT: THREE.MOUSE.PAN
};"""

content = content.replace(old_controls, new_controls)

# Update setMode
old_setmode = """    if (m === 'select' || m === 'eraser') {
        controls.mouseButtons = {
            LEFT: null,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN, // Default to PAN
        };
        controls.enabled = true;
        renderer.domElement.style.cursor = 'crosshair';
    } else {
        controls.mouseButtons = {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN,
        };
        controls.enabled = true;
        renderer.domElement.style.cursor = 'grab';
    }"""

new_setmode = """    if (m === 'select' || m === 'eraser') {
        controls.mouseButtons = {
            LEFT: -1, // Disable left-click for camera so we can paint
            MIDDLE: THREE.MOUSE.ZOOM,
            RIGHT: THREE.MOUSE.PAN, // Right-click pans by default in select mode
        };
        controls.enabled = true;
        renderer.domElement.style.cursor = 'crosshair';
    } else {
        controls.mouseButtons = {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.ZOOM,
            RIGHT: THREE.MOUSE.PAN, // Right-click pans by default in navigate mode
        };
        controls.enabled = true;
        renderer.domElement.style.cursor = 'grab';
    }"""

content = content.replace(old_setmode, new_setmode)

with open(target_file, 'w') as f:
    f.write(content)
