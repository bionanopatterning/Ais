#vertex
#version 420

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;

uniform mat4 vpMat;

out vec3 fnormal;

void main()
{
    gl_Position = vpMat * vec4(pos, 1.0);
    fnormal = normal;
}

#fragment
#version 420

in vec3 fnormal;
out vec4 fragColour;

uniform vec4 color;
uniform vec3 lightDir;
uniform vec3 viewDir;
uniform int style;
uniform float ambientStrength;
uniform float lightStrength;
uniform vec3 lightColour;

void main()
{
    if (style == 0) // Cartoon
    {
        float F_AMBIENT = ambientStrength;
        vec3 ambient = F_AMBIENT * color.rgb;
        vec3 diffuse = F_AMBIENT * color.rgb * max(0.0, dot(fnormal, viewDir));
        float d = dot(fnormal, viewDir);
        if (d < 0.03)
        {
            ambient *= -1.0;
        }
        fragColour = vec4(ambient + diffuse, color.a);
    }
    else if (style == 1) // Phong
    {
        float F_AMBIENT = ambientStrength;
        float F_DIFFUSE = lightStrength;
        float F_SPECULAR = 0.2f * lightStrength;
        float F_EMISSIVE = ambientStrength * 0.3f;

        float SPEC_POWER = 12.0f;

        vec3 ambient = F_AMBIENT * color.rgb;

        vec3 diffuse = max(0.0, dot(normalize(fnormal), lightDir)) * F_DIFFUSE * lightColour * color.rgb;

        vec3 viewDir = normalize(gl_FragCoord.xyz);
        vec3 reflDir = reflect(-lightDir, fnormal);
        float specIntensity = pow(min(1.0, max(dot(viewDir, reflDir), 0.0)), SPEC_POWER);
        vec3 specular = F_SPECULAR * specIntensity * vec3(1.0, 1.0, 1.0);

        vec3 emissive = dot(normalize(fnormal), viewDir) * F_EMISSIVE * color.rgb;
        fragColour = vec4(ambient + diffuse + specular + emissive, color.a);
    }
    else if (style == 2) // Flat
    {
        fragColour = vec4(color.rgb, color.a);
    }
    else if (style == 3)
    {
        fragColour = vec4(fnormal * 0.5f + 0.5f, 1.0f);
    }
}