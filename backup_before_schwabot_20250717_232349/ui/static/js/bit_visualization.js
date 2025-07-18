/**
 * Bit Visualization Engine
 * ========================
 * 
 * Advanced bit mapping visualization with Three.js integration
 * Handles smooth transitions from 4-bit to 64-bit with special phaser effects at 42-bit
 * Supports high-frequency operations with drift compensation
 */

class BitVisualizationEngine {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.canvasId = canvasId;
        
        // Three.js scene setup
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: this.canvas, 
            alpha: true, 
            antialias: true 
        });
        
        // Bit visualization state
        this.currentBitLevel = 16;
        this.targetBitLevel = 16;
        this.transitionProgress = 0.0;
        this.processingIntensity = 0.5;
        this.thermalInfluence = 0.0;
        
        // Particle systems
        this.bitParticles = [];
        this.connectionLines = [];
        this.processingWaves = [];
        
        // Animation properties
        this.animationId = null;
        this.startTime = Date.now();
        this.lastFrameTime = 0;
        this.frameRate = 60;
        this.adaptiveQuality = true;
        
        // Drift compensation
        this.driftAccumulation = { x: 0, y: 0 };
        this.driftThreshold = 0.1;
        
        // Performance tracking
        this.performanceMetrics = {
            frameRate: 60,
            renderTime: 16.67,
            particleCount: 0,
            connectionCount: 0
        };
        
        // Initialize visualization
        this.init();
    }
    
    init() {
        // Setup renderer
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setClearColor(0x000000, 0);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Setup camera
        this.camera.position.set(0, 0, 5);
        this.camera.lookAt(0, 0, 0);
        
        // Setup lights
        this.setupLighting();
        
        // Initialize particle systems
        this.initializeParticleSystems();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Start animation loop
        this.startAnimation();
        
        console.log('[PASS] BitVisualizationEngine initialized');
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Point lights for bit particles
        this.bitLights = [];
        for (let i = 0; i < 8; i++) {
            const light = new THREE.PointLight(0x00ff88, 0.5, 10);
            light.position.set(
                Math.cos(i * Math.PI / 4) * 3,
                Math.sin(i * Math.PI / 4) * 3,
                0
            );
            this.scene.add(light);
            this.bitLights.push(light);
        }
        
        // Directional light for overall illumination
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
    }
    
    initializeParticleSystems() {
        // Initialize with current bit level
        this.generateBitParticles(this.currentBitLevel);
        this.generateConnectionNetwork();
    }
    
    generateBitParticles(bitLevel) {
        // Clear existing particles
        this.bitParticles.forEach(particle => {
            this.scene.remove(particle.mesh);
        });
        this.bitParticles = [];
        
        // Generate particles based on bit level
        const particleCount = bitLevel;
        const radius = 2 + (bitLevel - 4) * 0.1;
        
        for (let i = 0; i < particleCount; i++) {
            const angle = (2 * Math.PI * i) / particleCount;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            const z = (Math.random() - 0.5) * 0.5;
            
            // Create particle geometry and material
            const geometry = new THREE.SphereGeometry(0.05, 8, 8);
            const material = new THREE.MeshPhongMaterial({
                color: this.getBitLevelColor(bitLevel),
                emissive: this.getBitLevelColor(bitLevel),
                emissiveIntensity: 0.3,
                transparent: true,
                opacity: 0.8
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(x, y, z);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            
            // Add to scene
            this.scene.add(mesh);
            
            // Store particle data
            this.bitParticles.push({
                mesh: mesh,
                originalPosition: { x, y, z },
                currentPosition: { x, y, z },
                velocity: { x: 0, y: 0, z: 0 },
                active: true,
                bitIndex: i,
                intensity: Math.random() * 0.5 + 0.5
            });
        }
        
        this.performanceMetrics.particleCount = particleCount;
    }
    
    generateConnectionNetwork() {
        // Clear existing connections
        this.connectionLines.forEach(connection => {
            this.scene.remove(connection.mesh);
        });
        this.connectionLines = [];
        
        // Generate connections based on processing intensity
        const connectionCount = Math.floor(this.bitParticles.length * this.processingIntensity);
        
        for (let i = 0; i < connectionCount; i++) {
            const startParticle = this.bitParticles[i];
            const endParticle = this.bitParticles[(i + 1) % this.bitParticles.length];
            
            if (startParticle && endParticle) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    startParticle.mesh.position,
                    endParticle.mesh.position
                ]);
                
                const material = new THREE.LineBasicMaterial({
                    color: this.getBitLevelColor(this.currentBitLevel),
                    transparent: true,
                    opacity: 0.6
                });
                
                const line = new THREE.Line(geometry, material);
                this.scene.add(line);
                
                this.connectionLines.push({
                    mesh: line,
                    startParticle: startParticle,
                    endParticle: endParticle,
                    strength: Math.random() * 0.5 + 0.5
                });
            }
        }
        
        this.performanceMetrics.connectionCount = connectionCount;
    }
    
    generateProcessingWaves() {
        // Special effects for phaser level (42-bit) and higher
        if (this.currentBitLevel >= 42) {
            // Clear existing waves
            this.processingWaves.forEach(wave => {
                this.scene.remove(wave.mesh);
            });
            this.processingWaves = [];
            
            // Generate wave geometry
            const waveCount = 3;
            
            for (let i = 0; i < waveCount; i++) {
                const geometry = new THREE.RingGeometry(1 + i * 0.5, 1.2 + i * 0.5, 32);
                const material = new THREE.MeshBasicMaterial({
                    color: 0x88ff00,
                    transparent: true,
                    opacity: 0.3,
                    side: THREE.DoubleSide
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.rotation.x = Math.PI / 2;
                mesh.position.z = -1;
                
                this.scene.add(mesh);
                
                this.processingWaves.push({
                    mesh: mesh,
                    originalScale: 1,
                    phase: i * Math.PI / 3,
                    speed: 0.02
                });
            }
        }
    }
    
    updateBitLevel(newBitLevel, transitionSpeed = 0.02) {
        if (newBitLevel !== this.targetBitLevel) {
            this.targetBitLevel = newBitLevel;
            this.transitionProgress = 0;
            
            // Update UI display
            this.updateBitLevelDisplay();
        }
        
        // Smooth transition
        if (this.currentBitLevel !== this.targetBitLevel) {
            this.transitionProgress = Math.min(1, this.transitionProgress + transitionSpeed);
            
            // Interpolate bit level
            this.currentBitLevel = this.lerp(
                this.currentBitLevel, 
                this.targetBitLevel, 
                this.transitionProgress
            );
            
            // Regenerate particles if transition is complete
            if (this.transitionProgress >= 1) {
                this.currentBitLevel = this.targetBitLevel;
                this.generateBitParticles(this.currentBitLevel);
                this.generateConnectionNetwork();
                
                // Generate processing waves for phaser level
                if (this.currentBitLevel >= 42) {
                    this.generateProcessingWaves();
                }
            }
        }
    }
    
    updateProcessingIntensity(intensity) {
        this.processingIntensity = Math.max(0, Math.min(1, intensity));
        
        // Update particle brightness and movement
        this.bitParticles.forEach(particle => {
            particle.intensity = this.processingIntensity;
            particle.mesh.material.emissiveIntensity = 0.3 * this.processingIntensity;
        });
        
        // Update connection strength
        this.connectionLines.forEach(connection => {
            connection.mesh.material.opacity = 0.6 * this.processingIntensity;
        });
        
        // Update processing intensity meter
        this.updateProcessingMeter();
    }
    
    updateThermalInfluence(thermalHealth) {
        this.thermalInfluence = 1 - thermalHealth; // Inverse relationship
        
        // Apply thermal color shifts
        const thermalColor = this.getThermalColor(thermalHealth);
        
        this.bitParticles.forEach(particle => {
            particle.mesh.material.color.setHex(thermalColor);
        });
        
        this.connectionLines.forEach(connection => {
            connection.mesh.material.color.setHex(thermalColor);
        });
    }
    
    handleHighFrequencyAllocations(allocations) {
        // Create temporary visual effects for high-frequency operations
        allocations.forEach((allocation, index) => {
            // Create burst effect
            this.createBurstEffect(
                allocation.position || { x: 0, y: 0, z: 0 },
                allocation.intensity || 0.5
            );
            
            // Apply drift compensation
            this.compensateDrift(allocation);
        });
    }
    
    createBurstEffect(position, intensity) {
        // Create temporary burst particles
        const burstParticles = new THREE.Group();
        const particleCount = Math.floor(intensity * 20);
        
        for (let i = 0; i < particleCount; i++) {
            const geometry = new THREE.SphereGeometry(0.02, 4, 4);
            const material = new THREE.MeshBasicMaterial({
                color: 0xffaa00,
                transparent: true,
                opacity: 0.8
            });
            
            const particle = new THREE.Mesh(geometry, material);
            particle.position.set(
                position.x + (Math.random() - 0.5) * 0.2,
                position.y + (Math.random() - 0.5) * 0.2,
                position.z + (Math.random() - 0.5) * 0.2
            );
            
            burstParticles.add(particle);
        }
        
        this.scene.add(burstParticles);
        
        // Animate burst and remove after completion
        const startTime = Date.now();
        const animateBurst = () => {
            const elapsed = Date.now() - startTime;
            const progress = elapsed / 1000; // 1 second duration
            
            if (progress < 1) {
                burstParticles.children.forEach(particle => {
                    particle.scale.setScalar(1 + progress * 2);
                    particle.material.opacity = 0.8 * (1 - progress);
                });
                requestAnimationFrame(animateBurst);
            } else {
                this.scene.remove(burstParticles);
            }
        };
        
        animateBurst();
    }
    
    compensateDrift(allocation) {
        // Accumulate drift from high-frequency operations
        this.driftAccumulation.x += (allocation.drift?.x || 0) * 0.1;
        this.driftAccumulation.y += (allocation.drift?.y || 0) * 0.1;
        
        // Apply compensation if drift exceeds threshold
        if (Math.abs(this.driftAccumulation.x) > this.driftThreshold ||
            Math.abs(this.driftAccumulation.y) > this.driftThreshold) {
            
            // Adjust camera position to compensate
            this.camera.position.x -= this.driftAccumulation.x * 0.1;
            this.camera.position.y -= this.driftAccumulation.y * 0.1;
            
            // Reset accumulation
            this.driftAccumulation.x *= 0.5;
            this.driftAccumulation.y *= 0.5;
        }
    }
    
    animate() {
        const currentTime = Date.now();
        const deltaTime = currentTime - this.lastFrameTime;
        this.lastFrameTime = currentTime;
        
        // Calculate frame rate
        this.frameRate = 1000 / deltaTime;
        this.performanceMetrics.frameRate = this.frameRate;
        
        // Adaptive quality adjustment
        if (this.adaptiveQuality) {
            this.adjustQualityBasedOnPerformance();
        }
        
        // Update particle animations
        this.updateParticleAnimations(currentTime);
        
        // Update connection animations
        this.updateConnectionAnimations(currentTime);
        
        // Update processing waves
        this.updateProcessingWaves(currentTime);
        
        // Update lights
        this.updateLighting(currentTime);
        
        // Render scene
        const renderStart = performance.now();
        this.renderer.render(this.scene, this.camera);
        this.performanceMetrics.renderTime = performance.now() - renderStart;
        
        // Continue animation
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    updateParticleAnimations(currentTime) {
        const time = currentTime * 0.001; // Convert to seconds
        
        this.bitParticles.forEach((particle, index) => {
            // Base orbital motion
            const angle = (2 * Math.PI * index) / this.bitParticles.length;
            const radius = 2 + (this.currentBitLevel - 4) * 0.1;
            
            // Add processing intensity effects
            const intensityEffect = this.processingIntensity * 0.5;
            const x = Math.cos(angle + time * 0.5) * radius + 
                     Math.sin(time * 2 + index) * intensityEffect;
            const y = Math.sin(angle + time * 0.5) * radius + 
                     Math.cos(time * 2 + index) * intensityEffect;
            const z = Math.sin(time + index) * 0.2 * intensityEffect;
            
            // Smooth position update
            particle.mesh.position.lerp(new THREE.Vector3(x, y, z), 0.1);
            
            // Rotation based on processing intensity
            particle.mesh.rotation.x += this.processingIntensity * 0.02;
            particle.mesh.rotation.y += this.processingIntensity * 0.01;
            
            // Scale pulsing for phaser level
            if (this.currentBitLevel >= 42) {
                const scale = 1 + Math.sin(time * 4 + index) * 0.3 * this.processingIntensity;
                particle.mesh.scale.setScalar(scale);
            }
        });
    }
    
    updateConnectionAnimations(currentTime) {
        const time = currentTime * 0.001;
        
        this.connectionLines.forEach(connection => {
            // Update line positions
            const positions = connection.mesh.geometry.attributes.position.array;
            positions[0] = connection.startParticle.mesh.position.x;
            positions[1] = connection.startParticle.mesh.position.y;
            positions[2] = connection.startParticle.mesh.position.z;
            positions[3] = connection.endParticle.mesh.position.x;
            positions[4] = connection.endParticle.mesh.position.y;
            positions[5] = connection.endParticle.mesh.position.z;
            
            connection.mesh.geometry.attributes.position.needsUpdate = true;
            
            // Animate opacity based on connection strength
            const opacity = 0.6 * connection.strength * 
                           (0.5 + 0.5 * Math.sin(time * 2 + connection.strength * 10));
            connection.mesh.material.opacity = opacity;
        });
    }
    
    updateProcessingWaves(currentTime) {
        const time = currentTime * 0.001;
        
        this.processingWaves.forEach(wave => {
            // Animate wave expansion
            const scale = wave.originalScale + Math.sin(time * 2 + wave.phase) * 0.3;
            wave.mesh.scale.setScalar(scale);
            
            // Animate opacity
            const opacity = 0.3 * (0.5 + 0.5 * Math.sin(time + wave.phase));
            wave.mesh.material.opacity = opacity;
            
            // Rotate waves
            wave.mesh.rotation.z += wave.speed;
        });
    }
    
    updateLighting(currentTime) {
        const time = currentTime * 0.001;
        
        // Animate bit lights
        this.bitLights.forEach((light, index) => {
            const intensity = 0.5 + 0.3 * Math.sin(time * 2 + index) * this.processingIntensity;
            light.intensity = intensity;
            
            // Color shift based on bit level
            light.color.setHex(this.getBitLevelColor(this.currentBitLevel));
        });
    }
    
    adjustQualityBasedOnPerformance() {
        // Reduce quality if frame rate drops
        if (this.frameRate < 30) {
            // Reduce particle count
            if (this.bitParticles.length > 16) {
                this.reduceBitParticles();
            }
            
            // Reduce connection count
            if (this.connectionLines.length > 8) {
                this.reduceConnections();
            }
        } else if (this.frameRate > 90) {
            // Increase quality if performance allows
            if (this.bitParticles.length < this.currentBitLevel) {
                this.generateBitParticles(this.currentBitLevel);
            }
        }
    }
    
    // Utility methods
    getBitLevelColor(bitLevel) {
        const colors = {
            4: 0x0088ff,   // Blue
            8: 0x00aaff,   // Light blue
            16: 0x00ccff,  // Cyan
            32: 0x00ffaa,  // Green-cyan
            42: 0x88ff00,  // Yellow-green (Phaser)
            64: 0xffaa00   // Orange
        };
        
        // Find closest bit level color
        const levels = Object.keys(colors).map(Number).sort((a, b) => a - b);
        const closestLevel = levels.reduce((prev, curr) => 
            Math.abs(curr - bitLevel) < Math.abs(prev - bitLevel) ? curr : prev
        );
        
        return colors[closestLevel];
    }
    
    getThermalColor(thermalHealth) {
        // Interpolate between thermal colors based on health
        if (thermalHealth > 0.8) return 0x00ff88; // Optimal
        if (thermalHealth > 0.6) return 0x88ff00; // Moderate
        if (thermalHealth > 0.4) return 0xffaa00; // Warm
        if (thermalHealth > 0.2) return 0xff4400; // Hot
        return 0xff0044; // Critical
    }
    
    lerp(start, end, t) {
        return start + (end - start) * t;
    }
    
    updateBitLevelDisplay() {
        const currentLevelElement = document.getElementById('current-bit-level');
        const targetLevelElement = document.getElementById('target-bit-level');
        const progressBar = document.getElementById('progress-bar');
        
        if (currentLevelElement) {
            currentLevelElement.textContent = Math.floor(this.currentBitLevel);
            currentLevelElement.setAttribute('data-level', Math.floor(this.currentBitLevel));
        }
        
        if (targetLevelElement) {
            targetLevelElement.textContent = `→ ${this.targetBitLevel}`;
        }
        
        if (progressBar) {
            progressBar.style.width = `${this.transitionProgress * 100}%`;
        }
    }
    
    updateProcessingMeter() {
        const intensityFill = document.getElementById('intensity-fill');
        const intensityValue = document.getElementById('intensity-value');
        
        if (intensityFill) {
            intensityFill.style.width = `${this.processingIntensity * 100}%`;
        }
        
        if (intensityValue) {
            intensityValue.textContent = `${Math.round(this.processingIntensity * 100)}%`;
        }
    }
    
    setupEventListeners() {
        // Handle window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = this.canvas.clientWidth / this.canvas.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        });
        
        // Handle visibility change for performance optimization
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseAnimation();
            } else {
                this.resumeAnimation();
            }
        });
    }
    
    startAnimation() {
        this.lastFrameTime = Date.now();
        this.animate();
    }
    
    pauseAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    resumeAnimation() {
        if (!this.animationId) {
            this.startAnimation();
        }
    }
    
    destroy() {
        this.pauseAnimation();
        
        // Clean up Three.js resources
        this.bitParticles.forEach(particle => {
            particle.mesh.geometry.dispose();
            particle.mesh.material.dispose();
            this.scene.remove(particle.mesh);
        });
        
        this.connectionLines.forEach(connection => {
            connection.mesh.geometry.dispose();
            connection.mesh.material.dispose();
            this.scene.remove(connection.mesh);
        });
        
        this.processingWaves.forEach(wave => {
            wave.mesh.geometry.dispose();
            wave.mesh.material.dispose();
            this.scene.remove(wave.mesh);
        });
        
        this.renderer.dispose();
        
        console.log('[PASS] BitVisualizationEngine destroyed');
    }
    
    getPerformanceMetrics() {
        return {
            ...this.performanceMetrics,
            driftAccumulation: { ...this.driftAccumulation },
            currentBitLevel: this.currentBitLevel,
            targetBitLevel: this.targetBitLevel,
            transitionProgress: this.transitionProgress,
            processingIntensity: this.processingIntensity
        };
    }
}

// Export for global use
window.BitVisualizationEngine = BitVisualizationEngine; 