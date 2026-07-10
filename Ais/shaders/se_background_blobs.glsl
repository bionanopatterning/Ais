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

#define MAXB 48

in vec2 uv;
out vec4 fragColour;

uniform int uN;
uniform int uShape;         // 0 = soft blob, 1 = brushstroke, 2 = disc (bokeh)
uniform vec2 uRes;          // screen size in px
uniform vec3 uBase;         // papery base colour
uniform float uIntensity;
uniform vec2 uPos[MAXB];    // centres in px (top-left origin)
uniform float uRad[MAXB];   // radius / circumradius in px
uniform vec3 uCol[MAXB];
uniform float uAng[MAXB];   // rotation (radians)
uniform float uAlp[MAXB];   // per-particle alpha (fade in/out)

void main()
{
    vec2 frag = vec2(uv.x, 1.0 - uv.y) * uRes;   // top-left origin, matches screen coords
    vec3 col = uBase;
    for (int i = 0; i < uN; i++)
    {
        float w;
        vec3 tgt = uCol[i];
        if (uShape == 2)
        {
            // soft disc (bokeh)
            float d = distance(frag, uPos[i]);
            float soft = max(6.0, uRad[i] * 0.30);
            w = smoothstep(uRad[i] + soft, uRad[i] - soft, d) * uIntensity * uAlp[i];
        }
        else if (uShape == 1)
        {
            // brushstroke: large, soft, elongated & rotated gaussian wisp
            vec2 rel = frag - uPos[i];
            float c = cos(uAng[i]);
            float s = sin(uAng[i]);
            vec2 lo = vec2(rel.x * c + rel.y * s, -rel.x * s + rel.y * c);
            float rr = max(uRad[i], 1.0);
            vec2 nrm = vec2(lo.x / (rr * 1.9), lo.y / (rr * 0.7));
            w = exp(-dot(nrm, nrm) * 1.6) * uIntensity * uAlp[i];
        }
        else
        {
            // soft gaussian blob (Aurora)
            float d = distance(frag, uPos[i]) / max(uRad[i], 1.0);
            w = exp(-d * d * 2.2) * uIntensity * uAlp[i];
        }
        col = mix(col, tgt, clamp(w, 0.0, 1.0));
    }
    fragColour = vec4(col, 1.0);
}
