import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class Viewer3D {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060b0c);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    this.camera.position.set(0, 0, 5);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.keyLight = new THREE.DirectionalLight(0x9deee0, 2.2);
    this.keyLight.position.set(3, 4, 5);
    this.scene.add(this.keyLight);
    this.scene.add(new THREE.AmbientLight(0x6a8c88, 0.8));
    this.mesh = null;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.animate();
  }

  animate() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.frame = requestAnimationFrame(() => this.animate());
  }

  resize() {
    const width = this.container.clientWidth || 1;
    const height = this.container.clientHeight || 1;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  resetCamera() {
    this.camera.position.set(0, 0, 5);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setDemoObject(enabled) {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.mesh = null;
    }
    if (enabled) {
      const geometry = new THREE.IcosahedronGeometry(1.15, 2);
      const material = new THREE.MeshStandardMaterial({ color: 0x70e0d1, roughness: 0.62, metalness: 0.08 });
      this.mesh = new THREE.Mesh(geometry, material);
      this.scene.add(this.mesh);
    }
  }

  setWireframe(enabled) {
    if (this.mesh) this.mesh.material.wireframe = enabled;
  }

  setOpacity(value) {
    if (this.mesh) {
      this.mesh.material.opacity = value;
      this.mesh.material.transparent = value < 1;
    }
  }

  setVisibility(visible) {
    if (this.mesh) this.mesh.visible = visible;
  }

  toggleFullscreen() {
    if (this.container.parentElement?.requestFullscreen) {
      this.container.parentElement.requestFullscreen();
    }
  }

  dispose() {
    cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
