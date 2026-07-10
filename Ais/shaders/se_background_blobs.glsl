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

#define MAXB 16

in vec2 uv;
out vec4 fragColour;

uniform int uN;
uniform int uShape;         // 0 = soft blob, 1 = confetti card, 2 = bokeh disc
uniform vec2 uRes;          // screen size in px
uniform vec3 uBase;         // papery base colour
uniform float uIntensity;
uniform vec2 uPos[MAXB];    // centres in px (top-left origin)
uniform float uRad[MAXB];   // radius / half-size in px
uniform vec3 uCol[MAXB];
uniform float uAng[MAXB];   // rotation (radians), for cards
uniform float uAlp[MAXB];   // per-particle alpha (fade in/out)

void main()
{
    vec2 frag = vec2(uv.x, 1.0 - uv.y) * uRes;   // top-left origin, matches screen coords
    vec3 col = uBase;
    for (int i = 0; i < uN; i++)
    {
        float w;
        if (uShape == 1)
        {
            // rounded confetti card: rotated rectangle with soft edges
            vec2 rel = frag - uPos[i];
            float c = cos(uAng[i]);
            float s = sin(uAng[i]);
            vec2 lo = vec2(rel.x * c + rel.y * s, -rel.x * s + rel.y * c);
            vec2 hf = vec2(uRad[i] * 1.5, uRad[i] * 0.62);
            vec2 e = abs(lo) - hf;
            float sd = min(max(e.x, e.y), 0.0) + length(max(e, vec2(0.0)));
            float soft = max(uRad[i] * 0.35, 8.0);
            w = smoothstep(soft, -soft, sd) * uIntensity * uAlp[i];
        }
        else if (uShape == 2)
        {
            // bokeh disc: fairly crisp circle
            float d = distance(frag, uPos[i]);
            float soft = max(uRad[i] * 0.25, 6.0);
            w = smoothstep(uRad[i] + soft, uRad[i] - soft, d) * uIntensity * uAlp[i];
        }
        else
        {
            // soft gaussian blob
            float d = distance(frag, uPos[i]) / max(uRad[i], 1.0);
            w = exp(-d * d * 2.2) * uIntensity * uAlp[i];
        }
        col = mix(col, uCol[i], clamp(w, 0.0, 1.0));
    }
    fragColour = vec4(col, 1.0);
}
