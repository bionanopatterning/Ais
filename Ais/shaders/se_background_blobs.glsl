#vertex
#version 420

layout(location = 0) in vec2 pos;

out vec2 uv;

void main()
{
    gl_Position = vec4(pos, 0.0, 1.0);
    uv = 0.5 + pos * 0.5;
}

#fragment
#version 420

#define MAXB 480

in vec2 uv;
out vec4 fragColour;

uniform int uN;
uniform int uShape;         // 0 = soft blob, 1 = brushstroke, 2 = disc (bokeh)
uniform vec2 uRes;          // screen size in px
uniform vec3 uBase;         // papery base colour
uniform float uIntensity;
// Per-blob data packed into two vec4s to stay under the fragment constant-
// register limit (5 separate arrays overflowed at high MAXB):
uniform vec4 uA[MAXB];      // xy = centre px, z = radius / half-width, w = alpha
uniform vec4 uB[MAXB];      // xyz = colour, w = angle / half-height

void main()
{
    vec2 frag = vec2(uv.x, 1.0 - uv.y) * uRes;   // top-left origin, matches screen coords
    vec3 col = uBase;
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
            // soft disc (bokeh)
            float d = distance(frag, pPos);
            float soft = max(6.0, pRad * 0.30);
            w = smoothstep(pRad + soft, pRad - soft, d) * uIntensity * pAlp;
        }
        else if (uShape == 1)
        {
            // brushstroke: large, soft, elongated & rotated gaussian wisp
            vec2 rel = frag - pPos;
            float c = cos(pAng);
            float s = sin(pAng);
            vec2 lo = vec2(rel.x * c + rel.y * s, -rel.x * s + rel.y * c);
            float rr = max(pRad, 1.0);
            vec2 nrm = vec2(lo.x / (rr * 1.9), lo.y / (rr * 0.7));
            w = exp(-dot(nrm, nrm) * 1.6) * uIntensity * pAlp;
        }
        else
        {
            // soft gaussian blob (Aurora)
            float d = distance(frag, pPos) / max(pRad, 1.0);
            w = exp(-d * d * 2.2) * uIntensity * pAlp;
        }
        col = mix(col, pCol, clamp(w, 0.0, 1.0));
    }
    fragColour = vec4(col, 1.0);
}
