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
uniform int uShape;         // 0 = soft blob, 1 = triangle, 2 = disc, 3 = lava hole
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
            // soft disc (bokeh)
            float d = distance(frag, uPos[i]);
            float soft = max(6.0, uRad[i] * 0.30);
            w = smoothstep(uRad[i] + soft, uRad[i] - soft, d) * uIntensity * uAlp[i];
        }
        else if (uShape == 3)
        {
            // lava hole: darken toward a deep tint of the colour (inverted "hole")
            float d = distance(frag, uPos[i]) / max(uRad[i], 1.0);
            w = exp(-d * d * 2.0) * uIntensity * uAlp[i];
            tgt = uCol[i] * 0.28;
        }
        else
        {
            // soft gaussian blob
            float d = distance(frag, uPos[i]) / max(uRad[i], 1.0);
            w = exp(-d * d * 2.2) * uIntensity * uAlp[i];
        }
        col = mix(col, tgt, clamp(w, 0.0, 1.0));
    }
    fragColour = vec4(col, 1.0);
}
