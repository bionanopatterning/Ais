#compute
#version 430

layout(binding = 0) uniform sampler2D overlay;
layout(binding = 1) uniform sampler2D depth_start;
layout(binding = 2) uniform sampler2D depth_stop;
layout(binding = 3, rgba32f) writeonly uniform image2D target;
layout(local_size_x = 16, local_size_y=16) in;

uniform mat4 ipMat;
uniform mat4 ivMat;
uniform mat4 pMat;
uniform mat4 vMat;
uniform vec2 viewportSize;
uniform float pixelSize;
uniform vec2 imgSize;
uniform float near;
uniform float far;
uniform float zLim;
uniform int style;
uniform float intensity;

float linearizeDepth(float depth)
{
    float z = depth; // Back to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));
}

bool isRayInVolume(vec3 ray)
{
    float x = abs(ray.x) / 1.3f;
    float y = abs(ray.y) / 1.3f;
    float z = abs(ray.z) / 1.3f;
    float rx = imgSize.x * pixelSize / 2.0f;
    float ry = imgSize.y * pixelSize / 2.0f;
    float rz = zLim;
    return (x <= rx && y <= ry && z < rz);
}

void main()
{
    // Get Ray start position.
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(px) / viewportSize;

    // Normalized device coordinates to world coordinates. First, find ndc's:
    float z_start = 2.0f * texture(depth_start, uv).r - 1.0f;
    float z_stop = 2.0f * texture(depth_stop, uv).r - 1.0f;
    float x = (2.0f * float(px.x)) / viewportSize.x - 1.0f;
    float y = (2.0f * float(px.y)) / viewportSize.y - 1.0f;

    vec4 cs_start = ipMat * vec4(x, y, z_start, 1.0);  // clip space (cs) start vec
    cs_start /= cs_start.w;
    vec4 cs_stop = ipMat * vec4(x, y, z_stop, 1.0);
    cs_stop /= cs_stop.w;

    vec3 start_pos = (ivMat * cs_start).xyz;
    vec3 stop_pos = (ivMat * cs_stop).xyz;
    vec3 dir = normalize(stop_pos - start_pos);

    if (z_start == 1.0f)
    {
        imageStore(target, px, vec4(0.0f, 0.0f, 0.0f, 1.0f));
    }
    else
    {
        vec3 pos = start_pos;
        vec2 uv = pos.xy / imgSize * 1.0f + 0.5f;
        vec4 rayValue = vec4(0.0f);
        bool rayInVolume = true;
        float pathLength = 0.0;
        float stepLength = length(dir);
        int MAX_ITER = 10000;
        float i = 0.0f;
        while (rayInVolume && i < MAX_ITER)
        {
            i += 1.0f;
            if (length(pos - start_pos) > length(stop_pos - start_pos))
            {
                break;
            }
            pos += dir;
            uv = pos.xy / imgSize * 1.0f + 0.5f;
            rayInVolume = isRayInVolume(pos);
            if (style == 0)
            {
                rayValue += texture(overlay, uv) * 2.0f;
            }
            else if (style == 1)
            {
                vec4 val = texture(overlay, uv);
                float threshold = intensity / 10.0f;
                if (val.r > threshold || val.g > threshold || val.b > threshold)
                {
                    rayValue = 1.0f * val / intensity;
                    break;
                }
            }
        }

        // Write to texture.
        float norm_fac_final = style == 0 ? 1 / 500.0f : 1.0f;
        imageStore(target, px, vec4(rayValue.xyz * norm_fac_final, 1.0f));
    }
}
