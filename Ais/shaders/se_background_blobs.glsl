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
uniform int uShape;         // 0 = soft blob, 1 = sharp triangle, 2 = crisp disc
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
        if (uShape == 1)
        {
            // sharp equilateral triangle (max of three half-planes), ~1px AA
            vec2 rel = frag - uPos[i];
            float c = cos(uAng[i]);
            float s = sin(uAng[i]);
            vec2 lo = vec2(rel.x * c + rel.y * s, -rel.x * s + rel.y * c);
            float rin = uRad[i] * 0.5;
            float sd = -1e9;
            for (int k = 0; k < 3; k++)
            {
                float th = 1.5707963 + float(k) * 2.0943951;
                sd = max(sd, dot(lo, vec2(cos(th), sin(th))) - rin);
            }
            w = smoothstep(1.2, -1.2, sd) * uIntensity * uAlp[i];
        }
        else if (uShape == 2)
        {
            // soft octagon (bokeh) - max of 8 half-planes, gentle edges
            vec2 rel = frag - uPos[i];
            float c = cos(uAng[i]);
            float s = sin(uAng[i]);
            vec2 lo = vec2(rel.x * c + rel.y * s, -rel.x * s + rel.y * c);
            float ap = uRad[i] * 0.92;
            float sd = -1e9;
            for (int k = 0; k < 8; k++)
            {
                float th = float(k) * 0.7853982;
                sd = max(sd, dot(lo, vec2(cos(th), sin(th))) - ap);
            }
            float soft = max(6.0, uRad[i] * 0.22);
            w = smoothstep(soft, -soft, sd) * uIntensity * uAlp[i];
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
