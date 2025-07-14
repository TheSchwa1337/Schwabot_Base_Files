//
// Cosine Similarity GPU Shader for Schwabot Trading System
// ========================================================
//
// Adaptive GLSL fragment shader for GPU-accelerated cosine similarity calculations
// Used for strategy matching in real-time trading operations
//
// Supports:
// - Pi 4 (VideoCore VI) through RTX 5090
// - Adaptive precision (mediump/highp)
// - Configurable matrix sizes (8x8 to 128x128)
// - Batch processing for multiple strategies
//

#version 300 es

// Precision directive - will be set programmatically based on GPU tier
// For Pi 4/low-tier: mediump float
// For mid/high-tier: highp float
precision PRECISION_PLACEHOLDER float;

// Input/Output
in vec2 v_texCoord;
out vec4 fragColor;

// Uniforms
uniform sampler2D u_tick_vector;          // Current market tick vector (1D texture)
uniform sampler2D u_strategy_matrix;      // Strategy vectors matrix (2D texture)
uniform int u_vector_length;              // Length of vectors being compared
uniform int u_strategy_index;             // Which strategy row to compare against
uniform int u_matrix_size;                // Matrix dimensions for bounds checking
uniform float u_epsilon;                  // Small value to prevent division by zero

// Configuration uniforms (set based on GPU DNA detection)
uniform bool u_enable_morphing;           // Enable advanced morphing operations
uniform int u_batch_size;                 // Number of strategies to process in parallel

// Constants
const float EPSILON_DEFAULT = 1e-8;
const float PI = 3.14159265359;

//
// Core cosine similarity computation
// cos(θ) = dot(A,B) / (||A|| * ||B||)
//
float computeCosineSimilarity(int vectorLength, int strategyIndex) {
    float dotProduct = 0.0;
    float normA = 0.0;  // Norm of tick vector
    float normB = 0.0;  // Norm of strategy vector
    
    // Iterate through vector components
    for (int i = 0; i < vectorLength; i++) {
        // Break if we exceed actual vector length (for dynamic sizing)
        if (i >= u_vector_length) break;
        
        // Sample tick vector value (horizontal texture, single row)
        float tickValue = texelFetch(u_tick_vector, ivec2(i, 0), 0).r;
        
        // Sample strategy vector value from specified row
        float strategyValue = texelFetch(u_strategy_matrix, ivec2(i, strategyIndex), 0).r;
        
        // Accumulate dot product and norms
        dotProduct += tickValue * strategyValue;
        normA += tickValue * tickValue;
        normB += strategyValue * strategyValue;
    }
    
    // Compute final cosine similarity with epsilon protection
    float denominator = sqrt(normA) * sqrt(normB) + u_epsilon;
    return dotProduct / denominator;
}

//
// Enhanced cosine similarity with morphing capabilities
// (Enabled on TIER_MID and above GPUs)
//
float computeEnhancedCosineSimilarity(int vectorLength, int strategyIndex) {
    float baseCosine = computeCosineSimilarity(vectorLength, strategyIndex);
    
    if (!u_enable_morphing) {
        return baseCosine;
    }
    
    // Apply morphing transformations for enhanced pattern detection
    // These operations help detect subtle pattern variations
    
    // 1. Sigmoid enhancement for non-linear pattern amplification
    float sigmoidEnhanced = 2.0 / (1.0 + exp(-4.0 * baseCosine)) - 1.0;
    
    // 2. Harmonic resonance calculation (for frequency-domain patterns)
    float harmonicWeight = sin(PI * baseCosine) * 0.1;
    
    // 3. Temporal morphing based on fragment coordinate (creates subtle variations)
    vec2 coord = gl_FragCoord.xy / vec2(float(u_matrix_size));
    float temporalMorph = sin(coord.x * PI * 2.0) * sin(coord.y * PI * 2.0) * 0.05;
    
    // Combine enhancements
    float enhancedSimilarity = baseCosine + harmonicWeight + temporalMorph;
    
    // Blend between base and enhanced based on similarity strength
    float blendFactor = smoothstep(0.3, 0.8, abs(baseCosine));
    return mix(baseCosine, enhancedSimilarity, blendFactor * 0.3);
}

//
// Batch processing for multiple strategies
// (Enabled on GPUs with sufficient parallel processing capability)
//
vec4 computeBatchCosineSimilarity() {
    vec4 results = vec4(0.0);
    int baseStrategyIndex = u_strategy_index;
    
    // Process up to 4 strategies in parallel (limited by vec4 output)
    for (int batch = 0; batch < min(u_batch_size, 4); batch++) {
        int currentIndex = baseStrategyIndex + batch;
        
        // Check bounds
        if (currentIndex >= u_matrix_size) {
            results[batch] = 0.0;
            continue;
        }
        
        // Compute similarity for this strategy
        if (u_enable_morphing) {
            results[batch] = computeEnhancedCosineSimilarity(u_vector_length, currentIndex);
        } else {
            results[batch] = computeCosineSimilarity(u_vector_length, currentIndex);
        }
    }
    
    return results;
}

//
// Adaptive precision optimization
// Adjusts computation precision based on detected GPU capabilities
//
float optimizeForPrecision(float value) {
    #ifdef MEDIUMP_PRECISION
        // For Pi 4 and low-tier GPUs: reduce precision to maintain performance
        return floor(value * 1000.0) / 1000.0;  // 3 decimal places
    #else
        // For high-tier GPUs: maintain full precision
        return value;
    #endif
}

//
// Main shader entry point
//
void main() {
    // Initialize output
    vec4 output = vec4(0.0, 0.0, 0.0, 1.0);
    
    // Check if we're within valid texture bounds
    vec2 texCoord = v_texCoord;
    if (texCoord.x < 0.0 || texCoord.x > 1.0 || texCoord.y < 0.0 || texCoord.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    // Batch processing mode (for higher-tier GPUs)
    if (u_batch_size > 1) {
        output = computeBatchCosineSimilarity();
    } 
    // Single strategy mode (for Pi 4 and low-tier GPUs)
    else {
        float similarity;
        if (u_enable_morphing) {
            similarity = computeEnhancedCosineSimilarity(u_vector_length, u_strategy_index);
        } else {
            similarity = computeCosineSimilarity(u_vector_length, u_strategy_index);
        }
        
        // Optimize precision based on GPU capabilities
        similarity = optimizeForPrecision(similarity);
        
        // Store result in red channel (cosine similarity range: [-1, 1])
        output = vec4(similarity, 0.0, 0.0, 1.0);
    }
    
    fragColor = output;
}

//
// Vertex shader companion (when needed)
//
/*
#vertex_shader
#version 300 es

in vec2 a_position;
in vec2 a_texCoord;

out vec2 v_texCoord;

void main() {
    v_texCoord = a_texCoord;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
*/ 