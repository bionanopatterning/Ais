from OpenGL.GL import *
from OpenGL.GLUT import *
#OpenGL.ERROR_CHECKING = False  # TODO: validate speed increase
from itertools import count
import numpy as np
import ctypes

"""All classes here are fairly straightforward implementations of standard OpenGL objects. Some notes:
in Texture/Framebuffer constructor there is a format parameter, which can be any of 'rgbu16', 'ru16', or 'rgba32f'.
For vertex and index buffers, the input vertices/indices data has to be a default python list [x1, y1, z1, x2, y2, ...] 
or [face11, face12, face13, face21, face22, etc.] (for vertex resp. index buffers)
All classes have bind() and unbind()* functions. (* no unbind for Texture). Texture.bind(slot) takes an argument, 
int:slot, to allow binding to any texture slot. When doing so, remember to set glActiveTexture(GL_TEXTURE0) later on 
whenever necessary.
"""


class Texture:
    idGenerator = count()

    IF_F_T = dict() # internalformat, format, type
    IF_F_T[None] = [GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT]
    IF_F_T["rgba32f"] = [GL_RGBA32F, GL_RGBA, GL_FLOAT]
    IF_F_T["rgb32f"] = [GL_RGB32F, GL_RGB, GL_FLOAT]
    IF_F_T["r32f"] = [GL_R32F, GL_RED, GL_FLOAT]
    IF_F_T["r8u"] = [GL_R8, GL_RED, GL_UNSIGNED_BYTE]
    IF_F_T["rg32f"] = [GL_R32F, GL_RG, GL_FLOAT]
    IF_F_T["rgbu16"] = [GL_RGB16UI, GL_RGB_INTEGER, GL_UNSIGNED_SHORT]
    IF_F_T["rgb8u"] = [GL_RGB8UI, GL_RGB_INTEGER, GL_UNSIGNED_INT]  # should be GL_BYTE but not touching it for now
    IF_F_T["rgba8u"] = [GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT]  # should be GL_BYTE but not touching it for now
    IF_F_T["rgba16u"] = [GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT]
    IF_F_T["rgba32u"] = [GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT]


    def __init__(self, format=None):
        if format:
            self.id = next(Texture.idGenerator)
            self.renderer_id = glGenTextures(1)
            self.width = 0
            self.height = 0
            if_f_t = Texture.IF_F_T[format]
            self.internalformat = if_f_t[0]
            self.format = if_f_t[1]
            self.type = if_f_t[2]
            self.wants_mipmap = False

            glBindTexture(GL_TEXTURE_2D, self.renderer_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    def bind(self, slot=0):
        glActiveTexture(GL_TEXTURE0 + int(slot))
        glBindTexture(GL_TEXTURE_2D, self.renderer_id)

    def update(self, pixeldata, width=None, height=None):
        self.bind()
        if width:
            self.width = width
        if height:
            self.height = height
        else:
            self.width = np.shape(pixeldata)[1]
            self.height = np.shape(pixeldata)[0]

        imgdata = None
        if not pixeldata is None:
            imgdata = pixeldata.flatten()
        glTexImage2D(GL_TEXTURE_2D, 0, self.internalformat, self.width, self.height, 0, self.format, self.type, imgdata)
        if self.wants_mipmap:
            glGenerateMipmap(GL_TEXTURE_2D)

    def update_subimage(self, pixeldata, xoffset, yoffset):
        self.bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, xoffset, yoffset, pixeldata.shape[1], pixeldata.shape[0], self.format, self.type, pixeldata.flatten())

    # def set_pixel(self, x, y, value):
    #     self.bind()
    #     glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, 1, 1, self.format, self.type, value)

    def set_linear_interpolation(self):
        glBindTexture(GL_TEXTURE_2D, self.renderer_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.wants_mipmap = False

    def set_linear_mipmap_interpolation(self):
        glBindTexture(GL_TEXTURE_2D, self.renderer_id)
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.wants_mipmap = True

    def set_no_interpolation(self):
        glBindTexture(GL_TEXTURE_2D, self.renderer_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        self.wants_mipmap = False


class Shader:
    """Uniforms can only be uploaded when Shaders is first manually bound by user."""
    def __init__(self, sourcecode = None):
        if sourcecode:
            self.shaderProgram = glCreateProgram()
            self.compile(sourcecode)

    def compile(self, sourcecode):
        shaderStages = dict()
        shaderObjects = list()
        currentStage = None
        # Parse source into individual shader stages' source code
        with open(sourcecode, 'r') as source:
            for line in source:
                if "#vertex" in line:
                    currentStage = GL_VERTEX_SHADER
                    shaderStages[currentStage] = ""
                elif "#fragment" in line:
                    currentStage = GL_FRAGMENT_SHADER
                    shaderStages[currentStage] = ""
                elif "#geometry" in line:
                    currentStage = GL_GEOMETRY_SHADER
                    shaderStages[currentStage] = ""
                elif "#compute" in line:
                    currentStage = GL_COMPUTE_SHADER
                    shaderStages[currentStage] = ""
                else:
                    shaderStages[currentStage] += line
        # Compile stages
        for key in shaderStages:
            shaderObjects.append(glCreateShader(key))
            glShaderSource(shaderObjects[-1], shaderStages[key])
            glCompileShader(shaderObjects[-1])
            status = glGetShaderiv(shaderObjects[-1], GL_COMPILE_STATUS)
            if status == GL_FALSE:
                if key == GL_VERTEX_SHADER:
                    strShaderType = "vertex"
                elif key == GL_FRAGMENT_SHADER:
                    strShaderType = "fragment"
                elif key == GL_GEOMETRY_SHADER:
                    strShaderType = "geometry"
                elif key == GL_COMPUTE_SHADER:
                    strShaderType = "compute"
                raise RuntimeError("Shaders compilation failure for type "+strShaderType+":\n" + glGetShaderInfoLog(shaderObjects[-1]).decode('utf-8'))
            glAttachShader(self.shaderProgram, shaderObjects[-1])
        glLinkProgram(self.shaderProgram)
        status = glGetProgramiv(self.shaderProgram, GL_LINK_STATUS)
        if status == GL_FALSE:
            raise RuntimeError("Shaders link failure:\n"+glGetProgramInfoLog(self.shaderProgram).decode('utf-8'))
        for shader in shaderObjects:
            glDetachShader(self.shaderProgram, shader)
            glDeleteShader(shader)

    def bind(self):
        glUseProgram(self.shaderProgram)

    def unbind(self):
        glUseProgram(0)

    def uniform1f(self, uniformName, uniformFloatValue):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniform1f(uniformLocation, uniformFloatValue)

    def uniform1i(self, uniformName, uniformIntValue):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniform1i(uniformLocation, uniformIntValue)

    def uniform2f(self, uniformName, uniformFloat2Value):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniform2f(uniformLocation, uniformFloat2Value[0], uniformFloat2Value[1])

    def uniform3f(self, uniformName, uniformFloat3Value):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniform3f(uniformLocation, uniformFloat3Value[0], uniformFloat3Value[1], uniformFloat3Value[2])

    def uniformmat4(self, uniformName, uniformMat4):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniformMatrix4fv(uniformLocation, 1, GL_TRUE, uniformMat4)

class VertexBuffer:
    """Not that vertices must be a default 1d python list. In __init__ it is cast into the required shape."""
    def __init__(self, vertex_data = None):
        self.vertexBufferObject = glGenBuffers(1)
        self.location = 0
        self.stride = 1
        if not vertex_data is None:
            self.update(vertex_data)

    def update(self, vertex_data):
        _data = np.asarray([[vertex_data]], dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)
        glBufferData(GL_ARRAY_BUFFER, _data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)

    def unbind(self):
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_location_and_stride(self, location, stride):
        self.location = location
        self.stride = stride
        glEnableVertexAttribArray(location)
        self.bind()
        glVertexAttribPointer(location, stride, GL_FLOAT, GL_FALSE, stride * 4, None)
        self.unbind()

    def set_divisor_to_per_instance(self):
        glVertexAttribDivisor(self.location, 1)


class IndexBuffer:
    """Note that indices must be a default python list. It is turned in to a np.array along the 2nd dimension with type np.uint16 before sending to GPU"""
    def __init__(self, indices):
        self.indexBufferObject = glGenBuffers(1)
        self.indices = np.asarray([indices], dtype = np.uint16)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBufferObject)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def bind(self):
        glBindBuffer(self.indexBufferObject)

    def unbind(self):
        glBindBuffer(0)

    def getCount(self):
        return self.indices.size


class VertexArray:
    def __init__(self, vertexBuffer = None, indexBuffer = None, attribute_format="xyzuv"):
        self.vertexBuffer = None
        self.indexBuffer = None
        self.initialized = False
        self.attribute_format = attribute_format
        self.vertexArrayObject = None
        if vertexBuffer:
            self.update(vertexBuffer, indexBuffer)

    def init(self):
        if self.initialized:
            return
        else:
            self.vertexArrayObject = glGenVertexArrays(1)
            self.initialized = True

    def update(self, vertexBuffer, indexBuffer):
        if not self.initialized:
            self.init()
        self.vertexBuffer = vertexBuffer
        self.indexBuffer = indexBuffer
        glBindVertexArray(self.vertexArrayObject)
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer.vertexBufferObject)
        if self.attribute_format == "xyzuv":
            glEnableVertexAttribArray(0)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.cast(0, ctypes.c_void_p))
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.cast(12, ctypes.c_void_p))
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.indexBufferObject)
        elif self.attribute_format == "xy":
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.cast(0, ctypes.c_void_p))
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.indexBufferObject)
        elif self.attribute_format == "xyuv":
            glEnableVertexAttribArray(0)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.cast(0, ctypes.c_void_p))
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.cast(8, ctypes.c_void_p))
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.indexBufferObject)
        elif self.attribute_format == "xyrgb":
            glEnableVertexAttribArray(0)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 20, ctypes.cast(0, ctypes.c_void_p))
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 20, ctypes.cast(8, ctypes.c_void_p))
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.indexBufferObject)
        glBindVertexArray(0)

    def bind(self):
        glBindVertexArray(self.vertexArrayObject)

    def unbind(self):
        glBindVertexArray(0)


class FrameBuffer:
    """For now, FrameBuffers have one texture (COLOR and DEPTH) only."""
    def __init__(self, width=None, height=None, texture_format="rgba32f"):
        """
        :param width: int
        :param height: int
        :param texture_format: format of the render texture. One of the texture formats allowed in Texture() (see above)
        """
        if width:
            # Set up internal parameters
            self.width = width
            self.height = height
            # Set up texture
            self.texture_format = texture_format
            self.texture = Texture(format=texture_format)
            self.texture.bind()
            self.texture.update(None, self.width, self.height)
            glBindTexture(GL_TEXTURE_2D, 0)
            # Set up depth render buffer
            self.depthRenderbuffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.depthRenderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)
            # Set up frame buffer
            self.framebufferObject = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture.renderer_id, 0)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depthRenderbuffer)

            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("Framebuffer binding failed, GPU might not support this configuration.")
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def clear(self, color):
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)
        glClearColor(color[0], color[1], color[2], 1.0 if len(color) == 3 else color[3])
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self, auto_set_viewport = True):
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)
        if auto_set_viewport:
            glViewport(0, 0, self.width, self.height)

    def unbind(self, viewport = None):
        """When the FBO was bound, the viewport may have been changed. Remember to change it back to the right size! Forgetting to do so does not cause errors, but subsequent viewports may appear clipped. Viewport:tuple(int, int, int, int)"""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if viewport:
            glViewport(*viewport)

    def __str__(self):
        return f"FrameBuffer object with colour and depth texture. Colour texture format is {self.texture_format}, size is {self.width} x {self.height}"