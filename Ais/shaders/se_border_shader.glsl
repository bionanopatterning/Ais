#vertex
#version 420

layout(location = 0) in vec2 xy;

uniform mat4 cameraMatrix;
uniform mat4 modelMatrix;
uniform float z_pos;
out vec2 fXY;

void main()
{
    gl_Position = cameraMatrix * modelMatrix * vec4(xy, z_pos, 1.0);
    vec2 fXY = xy;
}

#fragment
#version 420

#define MAXB 480

out vec4 fragmentColour;
in vec2 fXY;
uniform float alpha;

// The border is a hue chameleon: black by default, but wherever the living
// background field has colour behind it, it takes that colour vibrantly. The
// field is recomputed here from the same (packed) blob uniforms the background
// uses - two vec4s per blob to stay under the constant-register limit.
uniform int uN;             // 0 -> plain black border
uniform int uShape;         // 0 = soft blob, 1 = brushstroke, 2 = disc (bokeh), 3 = mosaic tile
uniform vec2 uRes;          // screen size in px
uniform float uIntensity;
uniform vec4 uA[MAXB];      // xy = centre px, z = radius / half-width, w = alpha
uniform vec4 uB[MAXB];      // xyz = colour, w = angle / half-height

void main()
{
    vec2 frag = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);   // top-left origin
    vec3 col = vec3(0.0);       // default black
    for (int i = 0; i < uN; i++)
    {
        vec2 pPos = uA[i].xy;
        float pRad = uA[i].z;
        float pAlp = uA[i].w;
        vec3 pCol = uB[i].xyz;
        float pAng = uB[i].w;
        float w;
        if (uShape == 2)
        {
            float d = distance(frag, pPos);
            float soft = max(6.0, pRad * 0.30);
            w = smoothstep(pRad + soft, pRad - soft, d) * uIntensity * pAlp;
        }
        else if (uShape == 1)
        {
            vec2 rel = frag - pPos;
            float c = cos(pAng);
            float s = sin(pAng);
            vec2 lo = vec2(rel.x * c + rel.y * s, -rel.x * s + rel.y * c);
            float rr = max(pRad, 1.0);
            vec2 nrm = vec2(lo.x / (rr * 1.9), lo.y / (rr * 0.7));
            w = exp(-dot(nrm, nrm) * 1.6) * uIntensity * pAlp;
        }
        else if (uShape == 3)
        {
            // mosaic tile: filled rectangle (half-width z, half-height w), hard edges
            vec2 rel = abs(frag - pPos);
            float inside = step(rel.x, pRad) * step(rel.y, pAng);
            w = inside * uIntensity * pAlp;
        }
        else
        {
            float d = distance(frag, pPos) / max(pRad, 1.0);
            w = exp(-d * d * 2.2) * uIntensity * pAlp;
        }
        // steeper than the background so even a faint wash lights the border up
        col = mix(col, pCol, clamp(w * 1.7, 0.0, 1.0));
    }
    fragmentColour = vec4(col, alpha);
}
