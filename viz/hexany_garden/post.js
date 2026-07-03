// Post stack: UnrealBloom for the glow, then a single "analog grade" pass —
// film grain, chromatic aberration, vignette, lifted blacks, and a slow
// gate-weave wobble. Grain is a pure hash of (uv, t): deterministic per frame.

import * as THREE from "three";
import { EffectComposer } from "../vendor/postprocessing/EffectComposer.js";
import { RenderPass } from "../vendor/postprocessing/RenderPass.js";
import { ShaderPass } from "../vendor/postprocessing/ShaderPass.js";
import { UnrealBloomPass } from "../vendor/postprocessing/UnrealBloomPass.js";

const GRADE_SHADER = {
  uniforms: {
    tDiffuse: { value: null },
    time: { value: 0 },
    grainAmount: { value: 0.055 },
    aberration: { value: 0.0003 },
    vignette: { value: 0.42 },
    lift: { value: 0.012 },
    weave: { value: 0.0006 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: `
    varying vec2 vUv;
    uniform sampler2D tDiffuse;
    uniform float time;
    uniform float grainAmount;
    uniform float aberration;
    uniform float vignette;
    uniform float lift;
    uniform float weave;

    float hash(vec2 p) {
      vec3 p3 = fract(vec3(p.xyx) * 0.1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }

    void main() {
      // Gate weave: the whole frame breathes very slightly.
      vec2 uv = vUv + vec2(
        sin(time * 0.31) * weave,
        cos(time * 0.23 + 1.3) * weave * 0.7
      );
      // Radial chromatic aberration.
      vec2 fromCenter = uv - 0.5;
      float r2 = dot(fromCenter, fromCenter);
      vec2 shift = fromCenter * r2 * aberration * 60.0;
      vec3 color;
      color.r = texture2D(tDiffuse, uv + shift).r;
      color.g = texture2D(tDiffuse, uv).g;
      color.b = texture2D(tDiffuse, uv - shift).b;
      // Lifted blacks: gentle haze floor.
      color = color * (1.0 - lift) + vec3(lift * 1.2, lift * 1.25, lift * 1.6);
      // Film grain, luminance-weighted toward shadows.
      float g = hash(uv * vec2(1913.0, 1723.0) + fract(time) * 91.7) - 0.5;
      float lum = dot(color, vec3(0.299, 0.587, 0.114));
      color += g * grainAmount * (1.0 - lum * 0.75);
      // Vignette.
      float vig = 1.0 - vignette * smoothstep(0.18, 0.85, r2 * 2.2);
      color *= vig;
      gl_FragColor = vec4(color, 1.0);
    }`,
};

export function createPost(renderer, scene, camera, width, height) {
  const composer = new EffectComposer(renderer);
  composer.setSize(width, height);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(width, height),
    0.85, // strength
    0.55, // radius
    0.38, // threshold: only genuine light sources bloom, never the sky
  );
  composer.addPass(bloom);
  const grade = new ShaderPass(GRADE_SHADER);
  composer.addPass(grade);

  function render(t, glow) {
    grade.uniforms.time.value = t;
    // The mix envelope breathes through bloom strength, gently.
    bloom.strength = 0.55 + glow * 0.45;
    composer.render();
  }

  return { render };
}
