import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

const BackgroundAnimation: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    const mountElement = mountRef.current;
    
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountElement.appendChild(renderer.domElement);

    // Neural network visualization
    const particles = new THREE.BufferGeometry();
    const particleCount = 500;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    // Create particle positions
    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 50;
      positions[i + 1] = (Math.random() - 0.5) * 50;
      positions[i + 2] = (Math.random() - 0.5) * 50;
      
      // Color gradient from blue to purple
      colors[i] = 0.2 + Math.random() * 0.3;
      colors[i + 1] = 0.3 + Math.random() * 0.4;
      colors[i + 2] = 0.8 + Math.random() * 0.2;
    }
    
    particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particles.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    // Particle material
    const particleMaterial = new THREE.PointsMaterial({
      size: 0.5,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.8,
    });
    
    const particleSystem = new THREE.Points(particles, particleMaterial);
    scene.add(particleSystem);
    
    // Connection lines
    const connectionGeometry = new THREE.BufferGeometry();
    const connectionPositions = new Float32Array(particleCount * particleCount * 6);
    let connectionIndex = 0;
    
    // Create connections between nearby particles
    for (let i = 0; i < particleCount; i++) {
      for (let j = i + 1; j < particleCount; j++) {
        const distance = Math.sqrt(
          Math.pow(positions[i * 3] - positions[j * 3], 2) +
          Math.pow(positions[i * 3 + 1] - positions[j * 3 + 1], 2) +
          Math.pow(positions[i * 3 + 2] - positions[j * 3 + 2], 2)
        );
        
        if (distance < 5 && Math.random() > 0.98) {
          connectionPositions[connectionIndex++] = positions[i * 3];
          connectionPositions[connectionIndex++] = positions[i * 3 + 1];
          connectionPositions[connectionIndex++] = positions[i * 3 + 2];
          connectionPositions[connectionIndex++] = positions[j * 3];
          connectionPositions[connectionIndex++] = positions[j * 3 + 1];
          connectionPositions[connectionIndex++] = positions[j * 3 + 2];
        }
      }
    }
    
    connectionGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(connectionPositions.slice(0, connectionIndex), 3)
    );
    
    const connectionMaterial = new THREE.LineBasicMaterial({
      color: 0x4080ff,
      opacity: 0.2,
      transparent: true,
      blending: THREE.AdditiveBlending,
    });
    
    const connections = new THREE.LineSegments(connectionGeometry, connectionMaterial);
    scene.add(connections);
    
    // Camera position
    camera.position.z = 30;
    
    // Animation
    const animate = () => {
      requestAnimationFrame(animate);
      
      // Rotate the entire system
      particleSystem.rotation.x += 0.0005;
      particleSystem.rotation.y += 0.001;
      connections.rotation.x += 0.0005;
      connections.rotation.y += 0.001;
      
      // Pulse effect
      const time = Date.now() * 0.001;
      particleMaterial.size = 0.5 + Math.sin(time) * 0.1;
      
      renderer.render(scene, camera);
    };
    
    animate();
    
    // Handle resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    
    window.addEventListener('resize', handleResize);
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountElement) {
        mountElement.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={mountRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: -1,
        background: 'radial-gradient(ellipse at center, #0a0e27 0%, #000000 100%)',
      }}
    />
  );
};

export default BackgroundAnimation; 