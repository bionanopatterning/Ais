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

#define MAXB 48

out vec4 fragmentColour;
in vec2 fXY;
uniform float alpha;

// The border is a hue chameleon: black by default, but wherever the living
// background field has colour behind it, it takes that colour vibrantly. The
// field is recomputed here from the same blob uniforms the background uses.
uniform int uN;             // 0 -> plain black border
uniform int uShape;         // 0 = soft blob, 1 = brushstroke, 2 = disc (bokeh)
uniform vec2 uRes;          // screen size in px
uniform float uIntensity;
uniform vec2 uPos[MAXB];
uniform float uRad[MAXB];
uniform vec3 uCol[MAXB];
uniform float uAng[MAXB];
uniform float uAlp[MAXB];

void main()
{
    vec2 frag = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);   // top-left origin
    vec3 col = vec3(0.0);       // default black
    for (int i = 0; i < uN; i++)
    {
        float w;
        if (uShape == 2)
        {
            float d = distance(frag, uPos[i]);
            float soft = max(6.0, uRad[i] * 0.30);
            w = smoothstep(uRad[i] + soft, uRad[i] - soft, d) * uIntensity * uAlp[i];
        }
        else if (uShape == 1)
        {
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
            float d = distance(frag, uPos[i]) / max(uRad[i], 1.0);
            w = exp(-d * d * 2.2) * uIntensity * uAlp[i];
        }
        // steeper than the background so even a faint wash lights the border up
        col = mix(col, uCol[i], clamp(w * 1.7, 0.0, 1.0));
    }
    fragmentColour = vec4(col, alpha);
}
