// creates a "star box" centered around the origin
// wasd + up/down movement available, along with a speedup
// movement is based on current camera direction

#include <iostream>
#include <list>
#include <vector>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <string>

#define GLM_ENABLE_EXPERIMENTAL
#define GLEW_STATIC 1                   // This allows linking with Static Library on Windows, without DLL
#include <GL/glew.h>                    // Include GLEW - OpenGL Extension Wrangler
#include <GLFW/glfw3.h>                 // cross-platform interface for creating a graphical context, initializing OpenGL and binding inputs
#include <glm/glm.hpp>                  // GLM is an optimized math library with syntax to similar to OpenGL Shading Language
#include <glm/gtc/matrix_transform.hpp> // include this to create transformation matrices
#include <glm/common.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <glm/gtx/transform.hpp>
#include "OBJloader.h"


#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"


using namespace glm;
using namespace std;

ma_engine gAudioEngine; // global audio engine

// Fire-and-forget helper for overlapping SFX.
inline void PlaySfx(const char* path, float volume = 1.0f) {
    if (!path || !path[0]) return;
    ma_engine_play_sound(&gAudioEngine, path, NULL);
    ma_engine_set_volume(&gAudioEngine, volume);
}

// SFX file paths (adjust as you like)
static const char* SFX_DOUBLE_BARREL = "Audio/double_barrel.wav";
static const char* SFX_GRENADE       = "Audio/grenade.wav";
static const char* SFX_SNIPER        = "Audio/sniper_laser.wav";
// ================================================================

// namespaces
using namespace glm;
using namespace std;

// star properties
const int NUM_STARS = 2000;
const float MAX_DISTANCE = 200.0f;
const float MIN_SIZE = 0.02f;
const float MAX_SIZE = 0.15f;
const float MOVE_SPEED = 8.0f;

// fixed window dimensions
const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 800;

// camera variables
glm::vec3 cameraPos = glm::vec3(0.0f, 20.0f, 50.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.3f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float yaw = -90.0f;
float pitch = -15.0f;
bool firstMouse = true;
float lastX = WINDOW_WIDTH / 2.0f;
float lastY = WINDOW_HEIGHT / 2.0f;

GLuint asteroidTexture;

// moons
struct Moon {
    vec3 offset;
    float radius;
    vec3 color;
    float orbitRadius;
    float orbitSpeed;
    float currentAngle;
};

struct Asteroid {
    vec3 position; //where
    vec3 rotationAxis; //spinny
    float rotationAngle; //more spinny
    float rotationSpeed; //more more spinny
    float orbitRadius; //keep in range of belt
    float orbitSpeed;
    float currentOrbitAngle;
    float size;
    int modelIndex; // 3 asteroid textures
};

vector<Model> asteroidModels;
vector<Asteroid> asteroids;

void loadAsteroidModels() {
    for (int i = 1; i <= 3; i++) {
        Model model;
        std::vector<glm::vec2> temp_uvs;
        if (loadOBJ(("models/asteroids/asteroid" + std::to_string(i) + ".obj").c_str(), 
                   model.vertices, model.normals, temp_uvs)) {
            model.texCoords = temp_uvs;  // Store texture coordinates
            
            // Normalize asteroid size
            float maxExtent = 0.0f;
            for (const auto& vertex : model.vertices) {
                maxExtent = std::max(maxExtent, std::abs(vertex.x));
                maxExtent = std::max(maxExtent, std::abs(vertex.y));
                maxExtent = std::max(maxExtent, std::abs(vertex.z));
            }
            
            if (maxExtent > 0.0f) {
                float scale = 1.0f / maxExtent;
                for (auto& vertex : model.vertices) {
                    vertex *= scale;
                }
            }
            
            model.setupBuffers();
            asteroidModels.push_back(model);
        } else {
            std::cerr << "Failed to load asteroid model " << i << std::endl;
        }
    }
}
// hassan    
// creates belt with asteroids
void generateAsteroidBelt(int count, float minRadius, float maxRadius) {
    for (int i = 0; i < count; i++) {
        Asteroid asteroid;
        asteroid.modelIndex = rand() % 3;
        
        // random orbit within belt radius
        asteroid.orbitRadius = minRadius + static_cast<float>(rand()) / 
                              (static_cast<float>(RAND_MAX/(maxRadius - minRadius)));
        asteroid.currentOrbitAngle = static_cast<float>(rand() % 360);
        asteroid.orbitSpeed = 0.0005f + static_cast<float>(rand()) / 
                             (static_cast<float>(RAND_MAX/(0.002f)));
        
        float y = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 4.0f - 2.0f);
        
        float x = asteroid.orbitRadius * cos(radians(asteroid.currentOrbitAngle));
        float z = asteroid.orbitRadius * sin(radians(asteroid.currentOrbitAngle));
        asteroid.position = vec3(x, y, z);
        
        // random rotation sppeed and axis
        asteroid.rotationAxis = normalize(vec3(
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f
        ));
        asteroid.rotationAngle = static_cast<float>(rand() % 360);
        asteroid.rotationSpeed = static_cast<float>(rand()) / RAND_MAX * 0.5f;
        asteroid.size = 0.1f + static_cast<float>(rand()) / 
                       (static_cast<float>(RAND_MAX/(0.4f)));
        
        asteroids.push_back(asteroid);
    }
}
// hassan    
// keeps them spinning
void updateAsteroids(float deltaTime) {
    for (auto& asteroid : asteroids) {
        asteroid.currentOrbitAngle += asteroid.orbitSpeed * deltaTime * 60.0f;
        float x = asteroid.orbitRadius * cos(radians(asteroid.currentOrbitAngle));
        float z = asteroid.orbitRadius * sin(radians(asteroid.currentOrbitAngle));
        asteroid.position.x = x;
        asteroid.position.z = z;
        asteroid.rotationAngle += asteroid.rotationSpeed * deltaTime * 60.0f;
    }
}

// hassan    
void renderAsteroids(const mat4& view, const mat4& projection, GLuint shader) {
    glUseProgram(shader);
    
    // give view and projection matrices to shader
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, &projection[0][0]);
    
    // give sun positions for lighting
    vec3 sunPositions[2];
    vec3 sunColors[2];
    int sunIdx = 0;
    for (auto it = planets.begin(); it != planets.end() && sunIdx < 2; ++it, ++sunIdx) {
        sunPositions[sunIdx] = it->position;
        sunColors[sunIdx] = it->color;
    }
    glUniform3fv(glGetUniformLocation(shader, "sunPositions"), 2, &sunPositions[0][0]);
    glUniform3fv(glGetUniformLocation(shader, "sunColors"), 2, &sunColors[0][0]);
    glUniform3fv(glGetUniformLocation(shader, "viewPos"), 1, &cameraPos[0]);
    
    
    GLint modelLoc = glGetUniformLocation(shader, "model");
    GLint useTextureLoc = glGetUniformLocation(shader, "useTexture");
    GLint isSunLoc = glGetUniformLocation(shader, "isSun");
    GLint colorLoc = glGetUniformLocation(shader, "color");
    
    for (const auto& asteroid : asteroids) {
        mat4 model = mat4(1.0f);
        model = translate(model, asteroid.position);
        model = rotate(model, radians(asteroid.rotationAngle), asteroid.rotationAxis);
        model = scale(model, vec3(asteroid.size));
        
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model[0][0]);
        glUniform1i(useTextureLoc, 0);
        glUniform1i(isSunLoc, 0);
        
        // set gray color 
        vec3 grayColor;
        switch(asteroid.modelIndex) {
            case 0: grayColor = vec3(0.4f, 0.4f, 0.4f); break; // Dark gray
            case 1: grayColor = vec3(0.6f, 0.6f, 0.6f); break; // Medium gray  
            case 2: grayColor = vec3(0.5f, 0.5f, 0.5f); break; // Light gray
            default: grayColor = vec3(0.5f, 0.5f, 0.5f); break;
        }
        glUniform3f(colorLoc, grayColor.x, grayColor.y, grayColor.z);
        
        // Draw
        if (asteroid.modelIndex < asteroidModels.size()) {
            asteroidModels[asteroid.modelIndex].Draw();
        }
    }
}

// solar system parameters
struct Planet {
    vec3 position;
    float radius;
    vec3 color;
    float orbitRadius;
    float orbitSpeed;
    float currentAngle;
    float rotationAngle;
    float rotationSpeed;
    list<Moon> moons;

    bool hasRing = false;
    float ringInnerRadius = 0.0f;
    float ringOuterRadius = 0.0f;
    vec3 ringColor = vec3(1.0f);
};

// planet variables
list<Planet> planets;
float sunRadius = 5.0f;

struct Model {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    GLuint vao, vbo[2]; // 0: vertices, 1: normals

    void setupBuffers() {
        glGenVertexArrays(1, &vao);
        glGenBuffers(2, vbo);
        glBindVertexArray(vao);

        // Vertices
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

        // Normals
        if (!normals.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
            glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), normals.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        }

        glBindVertexArray(0);
    }

    void Draw() const {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
        glBindVertexArray(0);
    }
};

struct Satellite {
    vec3 position;
    float orbitRadius;
    float orbitSpeed;
    float currentAngle;
    float tiltAngle;
    float rotationAngle;
    float rotationSpeed;
    Model model;
    float size;
};

// -------- Projectiles & Muzzle Flashes --------
struct Projectile {
    glm::vec3 pos;
    glm::vec3 vel;
    float radius;
    float life;    // seconds
    bool grenade;  // true = grenade (gravity), false = bullet
};
struct MuzzleFlash {
    glm::vec3 pos;
    glm::vec3 right;
    glm::vec3 up;
    float life;    // seconds
};

// ==================== Sniper scope, laser, targets, score ====================
bool gScoped = false;
bool gScopeKeyHeld = false;     // edge-trigger for 'C'
bool gLaserKeyHeld = false;     // not strictly needed here

float gCurrentFov = 45.0f;
const float FOV_NORMAL = 45.0f;
const float FOV_SCOPED = 15.0f;
const float FOV_SMOOTH = 12.0f;   // larger = faster easing

// Laser projectile
const float LASER_SPEED = 2000.0f; // units per second
const float LASER_RADIUS = 2.5f;
const float LASER_LIFETIME = 2.0f;
float laserCooldown = 0.0f;
const float LASER_COOLDOWN = 0.18f;

int gScore = 0;

// Extend Projectile to mark laser type
struct ProjectileEx : Projectile {
    bool laser;
    ProjectileEx(glm::vec3 P, glm::vec3 V, float R, float L, bool G, bool LZ) {
        pos=P; vel=V; radius=R; life=L; grenade=G; laser=LZ;
    }
};

// Targets floating near planets
struct TargetBox {
    int planetIndex;       // which planet (world pos follows that planet)
    glm::vec3 offset;      // local offset around that planet
    float size;            // half-extent (cube)
    bool lasting;
};

// -------- Gun cube geometry --------
float gunVertices[] = {
    // positions
    -0.05f, -0.05f,  0.0f,
     0.05f, -0.05f,  0.0f,
     0.05f,  0.05f,  0.0f,
    -0.05f,  0.05f,  0.0f,
    -0.05f, -0.05f,  0.5f,
     0.05f, -0.05f,  0.5f,
     0.05f,  0.05f,  0.5f,
    -0.05f,  0.05f,  0.5f
};
unsigned int gunIndices[] = {
    0,1,2, 2,3,0,  4,5,6, 6,7,4,  0,3,7, 7,4,0,
    1,5,6, 6,2,1,  3,2,6, 6,7,3,  0,4,5, 5,1,0
};

// -------- Ship unit-cube --------
unsigned int shipVAO=0, shipVBO=0, shipEBO=0;
float shipVertices[] = {
    -0.5f,-0.5f,-0.5f,  0.5f,-0.5f,-0.5f,  0.5f, 0.5f,-0.5f, -0.5f, 0.5f,-0.5f,
    -0.5f,-0.5f, 0.5f,  0.5f,-0.5f, 0.5f,  0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f
};
unsigned int shipIndices[] = {
    0,1,2, 2,3,0,  4,5,6, 6,7,4,  0,3,7, 7,4,0,
    1,5,6, 6,2,1,  3,2,6, 6,7,3,  0,4,5, 5,1,0
};

// -------- Muzzle flash billboard quad (pos+uv) --------
unsigned int quadVAO=0, quadVBO=0, quadEBO=0;
float quadVerts[] = {
    // x,y,z   u,v
    -0.5f,-0.5f,0.0f,   0.0f,0.0f,
     0.5f,-0.5f,0.0f,   1.0f,0.0f,
     0.5f, 0.5f,0.0f,   1.0f,1.0f,
    -0.5f, 0.5f,0.0f,   0.0f,1.0f
};
unsigned int quadIdx[] = {0,1,2, 2,3,0};

// -------- Targets & scope resources --------
std::vector<ProjectileEx> projectiles;
std::vector<MuzzleFlash> flashes;

std::vector<TargetBox> targets;
GLuint targetVAO=0, targetVBO=0, targetEBO=0;

GLuint scopeVAO=0, scopeVBO=0, scopeEBO=0, scopeProg=0;

vector<Satellite> satellites;
GLuint satelliteShader;

// shaders with dynamic lighting for two suns
const char *vertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in float aSize;
    layout (location = 2) in vec2 aTexCoords;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragPos;
    out vec2 TexCoord;
    out vec3 Normal;

    void main()
    {
        TexCoord = aTexCoords;

        vec4 worldPos = model * vec4(aPos, 1.0);
        FragPos = worldPos.xyz;
        Normal = normalize(mat3(transpose(inverse(model))) * aPos);
        gl_Position = projection * view * worldPos;
        gl_PointSize = aSize * 20.0;
    }
)glsl";

const char *fragmentShaderSource = R"glsl(
    #version 330 core

    uniform sampler2D planetTexture;
    uniform int useTexture;
    uniform vec3 color;
    uniform int isSun;

    uniform vec3 sunPositions[2];
    uniform vec3 sunColors[2];
    uniform vec3 viewPos;

    uniform float uAlpha; // NEW

    in vec2 TexCoord;
    in vec3 FragPos;
    in vec3 Normal;
    out vec4 FragColor;

    void main()
    {
        vec4 baseColor = (useTexture == 1) ? texture(planetTexture, TexCoord) : vec4(color, uAlpha);

        if (isSun == 1) {
            FragColor = vec4(baseColor.rgb, uAlpha);
            return;
        }

        vec3 norm = normalize(Normal);
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 ambient = 0.15 * baseColor.rgb;
        vec3 lighting = ambient;

        for (int i = 0; i < 2; ++i) {
            vec3 lightDir = normalize(sunPositions[i] - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            float spec = 0.0;
            if (diff > 0.0) {
                vec3 reflectDir = reflect(-lightDir, norm);
                spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
            }
            vec3 diffuse = diff * sunColors[i] * baseColor.rgb;
            vec3 specular = 0.3 * spec * sunColors[i];
            lighting += diffuse + specular;
        }
        FragColor = vec4(lighting, uAlpha);
    }
)glsl";

// Satellite shaders
const char *satelliteVertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragPos;
    out vec3 Normal;

    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)glsl";

const char *satelliteFragmentShaderSource = R"glsl(
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;

    out vec4 FragColor;

    uniform vec3 sunPositions[2];
    uniform vec3 sunColors[2];
    uniform vec3 viewPos;
    uniform vec3 objectColor;

    void main() {
        vec3 norm = normalize(Normal);
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 ambient = vec3(0.1);
        vec3 lighting = ambient;

        for (int i = 0; i < 2; i++) {
            vec3 lightDir = normalize(sunPositions[i] - FragPos);
            float diff = max(dot(norm, lightDir), 0.2);
            vec3 diffuse = diff * sunColors[i];

            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = 0.3 * spec * sunColors[i];

            lighting += (diffuse + specular);
        }

        FragColor = vec4(lighting * vec3(0.1411764706,0.1411764706,0.1411764706), 1.0);
    }
)glsl";

// ==================== Scope overlay shaders ====================
const char* scopeVS = R"glsl(
#version 330 core
layout (location=0) in vec2 aPos;   // NDC
layout (location=1) in vec2 aUV;
out vec2 vUV;
void main(){
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)glsl";

const char* scopeFS = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform float aspect;    // width/height
uniform float alpha;     // outside darkness alpha (0..1)
uniform int drawCross;   // bool
uniform float lineW;     // crosshair line half width in UV

void main(){
    // Map to centered coords: x in [-1,1], y in [-1,1]
    vec2 p = vUV*2.0 - 1.0;
    p.x *= aspect;

    float r = length(p);
    float mask = step(r, 0.85);     // circle radius ~0.85

    // dark outside
    vec3 col = vec3(0.0);
    float a = (1.0 - mask) * alpha;

    // crosshair lines inside the circle
    if (mask > 0.5 && drawCross==1) {
        float vLine = smoothstep(lineW, 0.0, abs(p.x));
        float hLine = smoothstep(lineW, 0.0, abs(p.y));
        float cross = max(vLine, hLine);
        col = mix(col, vec3(1.0), 1.0); // white lines
        a = max(a, cross*0.8); // show lines
    }

    FragColor = vec4(col, a);
}
)glsl";

GLuint shaderProgram;
GLuint sphereVAO, sphereVBO;          // planet array/buffer
GLuint starVAO, starVBO, starSizeVBO; // star array/buffer
list<vec3> starPositions;
list<float> starSizes;

GLuint sphereEBO;              // element buffer object for sphere indices
unsigned int sphereIndexCount; // indices for the sphere

// rings
unsigned int ringVAO, ringVBO, ringEBO;
int ringIndexCount;

// mouse movement
void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = (float)xpos;
        lastY = (float)ypos;
        firstMouse = false;
    }
    float xoffset = (float)xpos - lastX;
    float yoffset = lastY - (float)ypos;
    lastX = (float)xpos;
    lastY = (float)ypos;

    float sensitivity = 0.005f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    vec3 front;
    front.x = cos(radians(yaw)) * cos(radians(pitch));
    front.y = sin(radians(pitch));
    front.z = sin(radians(yaw)) * cos(radians(pitch));
    cameraFront = normalize(front);
}

// for keyboard movement
void processInput(GLFWwindow *window, float deltaTime) {
    float cameraSpeed = MOVE_SPEED * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= normalize(cross(cameraFront, cameraUp)) * cameraSpeed;

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += normalize(cross(cameraFront, cameraUp)) * cameraSpeed;

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraUp;

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraUp;
}

// creates random stars (unordered)
void generateStarField() {
    for (int i = 0; i < NUM_STARS; i++) {
        float distance = MAX_DISTANCE * pow(static_cast<float>(rand()) / RAND_MAX, 0.25f);
        float theta = 2.0f * pi<float>() * static_cast<float>(rand()) / RAND_MAX;
        float phi = acos(2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);

        float x = distance * sin(phi) * cos(theta);
        float y = distance * sin(phi) * sin(theta);
        float z = distance * cos(phi);

        starPositions.push_back(vec3(x, y, z));

        float size = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * pow(static_cast<float>(rand()) / RAND_MAX, 4.0f);
        starSizes.push_back(size);
    }
}

// generates sphere for planets, suns, moons
void createSphere(float radius, int sectors, int stacks) {
    int vertexCount = (stacks + 1) * (sectors + 1);
    int indexCount = stacks * sectors * 6;
    float *vertices = new float[vertexCount * 5]; // 3 pos + 2 UV
    unsigned int *indices = new unsigned int[indexCount];

    float sectorStep = 2 * glm::pi<float>() / sectors;
    float stackStep = glm::pi<float>() / stacks;

    int vertexIndex = 0;
    for (int i = 0; i <= stacks; ++i) {
        float stackAngle = glm::pi<float>() / 2 - i * stackStep;
        float xy = radius * cosf(stackAngle);
        float z = radius * sinf(stackAngle);

        for (int j = 0; j <= sectors; ++j) {
            float sectorAngle = j * sectorStep;
            float x = xy * cosf(sectorAngle);
            float y = xy * sinf(sectorAngle);

            vertices[vertexIndex++] = x;
            vertices[vertexIndex++] = y;
            vertices[vertexIndex++] = z;

            vertices[vertexIndex++] = (float)j / sectors;
            vertices[vertexIndex++] = (float)i / stacks;
        }
    }

    int index = 0;
    for (int i = 0; i < stacks; ++i) {
        int k1 = i * (sectors + 1);
        int k2 = k1 + sectors + 1;

        for (int j = 0; j < sectors; ++j, ++k1, ++k2) {
            if (i != 0) {
                indices[index++] = k1;
                indices[index++] = k2;
                indices[index++] = k1 + 1;
            }
            if (i != (stacks - 1)) {
                indices[index++] = k1 + 1;
                indices[index++] = k2;
                indices[index++] = k2 + 1;
            }
        }
    }

    sphereIndexCount = index;

    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);

    glBindVertexArray(sphereVAO);

    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, vertexCount * 5 * sizeof(float), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndexCount * sizeof(unsigned int), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    delete[] vertices;
    delete[] indices;
}

void loadSatelliteModels() {
    Satellite tess, quikscat;
    std::vector<glm::vec2> temp_uvs;

    if (!loadOBJ("models/TESS.obj", tess.model.vertices, tess.model.normals, temp_uvs)) {
        std::cerr << "Failed to load TESS model" << std::endl;
    }
    tess.model.setupBuffers();
    tess.orbitRadius = 1.0f;
    tess.orbitSpeed = 0.1f;
    tess.currentAngle = 0.0f;
    tess.tiltAngle = 30.0f;
    tess.size = 0.005f;
    tess.rotationAngle = 0.5f;
    tess.rotationSpeed = 0.025f;

    temp_uvs.clear();

    if (!loadOBJ("models/QuikSCAT.obj", quikscat.model.vertices, quikscat.model.normals, temp_uvs)) {
        std::cerr << "Failed to load QuikSCAT model" << std::endl;
    }
    quikscat.model.setupBuffers();
    quikscat.orbitRadius = 1.0f;
    quikscat.orbitSpeed = 0.1f;
    quikscat.currentAngle = 180.0f;
    quikscat.tiltAngle = -15.0f;
    quikscat.size = 0.025f;
    quikscat.rotationAngle = -180.0f;
    quikscat.rotationSpeed = 0.075f;

    temp_uvs.clear();

    satellites.push_back(tess);
    satellites.push_back(quikscat);
}

void renderSatellites(const mat4 &view, const mat4 &projection, GLuint satelliteShader) {
    glUseProgram(satelliteShader);

    // Find Earth's position
    vec3 earthPos;
    int planetIndex = 0;
    for (const auto &planet : planets) {
        if (planetIndex == 3) { // Earth is 4th planet (index 3)
            earthPos = planet.position;
            break;
        }
        planetIndex++;
    }

    // Sun positions and colors
    vec3 sunPositions[2];
    vec3 sunColors[2];
    planetIndex = 0;
    int sunCount = 0;
    for (const auto &planet : planets) {
        if (sunCount >= 2) break;
        if (planetIndex < 2) {
            sunPositions[sunCount] = planet.position;
            sunColors[sunCount] = planet.color;
            sunCount++;
        }
        planetIndex++;
    }

    glUniform3fv(glGetUniformLocation(satelliteShader, "sunPositions"), 2, &sunPositions[0][0]);
    glUniform3fv(glGetUniformLocation(satelliteShader, "sunColors"), 2, &sunColors[0][0]);
    glUniform3fv(glGetUniformLocation(satelliteShader, "viewPos"), 1, &cameraPos[0]);
    glUniform3f(glGetUniformLocation(satelliteShader, "objectColor"), 0.7f, 0.7f, 0.7f);

    for (const auto& sat : satellites) {
        mat4 model = mat4(1.0f);
        model = translate(model, sat.position);

        // Face toward Earth
        vec3 toEarth = normalize(earthPos - sat.position);
        vec3 up = vec3(0.0f, 1.0f, 0.0f);
        vec3 right = normalize(cross(toEarth, up));
        up = normalize(cross(right, toEarth));
        mat4 faceEarth = mat4(vec4(right, 0), vec4(up, 0), vec4(-toEarth, 0), vec4(0, 0, 0, 1));

        mat4 rotation = rotate(-radians(sat.rotationAngle), vec3(0, 1, 0));

        model = model * faceEarth * rotation;
        model = scale(model, vec3(sat.size));

        glUniformMatrix4fv(glGetUniformLocation(satelliteShader, "model"), 1, GL_FALSE, &model[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(satelliteShader, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(satelliteShader, "projection"), 1, GL_FALSE, &projection[0][0]);

        sat.model.Draw();
    }
}

// initial twin star system positions
void initSolarSystem() {
    float sunOrbitRadius = 12.0f;
    float sun1Angle = 0.0f;
    float sun2Angle = 180.0f;
    float sunOrbitSpeed = 0.18f;

    // sun 1
    planets.push_back({
        vec3(sunOrbitRadius, 0.0f, 0.0f),
        sunRadius,
        vec3(1.0f, 0.8f, 0.0f),
        sunOrbitRadius,
        sunOrbitSpeed,
        sun1Angle,
        0.0f,
        0.0f,
        {},
        false
    });
    // sun 2
    planets.push_back({
        vec3(-sunOrbitRadius, 0.0f, 0.0f),
        sunRadius,
        vec3(1.0f, 0.8f, 0.0f),
        sunOrbitRadius,
        sunOrbitSpeed,
        sun2Angle,
        0.0f,
        0.0f,
        {},
        false
    });

    planets.push_back({vec3(18.0f, 0.0f, 0.0f), 0.4f, vec3(0.7f, 0.7f, 0.7f), 18.0f, 0.04f, 0.0f});
    planets.push_back({vec3(22.0f, 0.0f, 0.0f), 0.6f, vec3(0.9f, 0.6f, 0.2f), 22.0f, 0.015f, 45.0f});
    planets.push_back({vec3(27.0f, 0.0f, 0.0f), 0.7f, vec3(0.2f, 0.4f, 0.9f), 27.0f, 0.01f, 90.0f});
    planets.push_back({vec3(32.0f, 0.0f, 0.0f), 0.5f, vec3(0.8f, 0.3f, 0.1f), 32.0f, 0.008f, 135.0f});
    planets.push_back({
        vec3(30.0f, 0.0f, 0.0f),
        1.5f,
        vec3(0.8f, 0.6f, 0.4f),
        42.0f,
        0.002f,
        180.0f});
    planets.push_back({
        vec3(50.0f, 0.0f, 0.0f),
        1.2f,
        vec3(0.9f, 0.8f, 0.5f),
        50.0f,
        0.0009f,
        225.0f,
        0.0f,
        0.2f,
        {},
        true,
        0.9f,
        1.3f,
        vec3(0.6f, 0.7f, 0.9f)
    });
    planets.push_back({vec3(60.0f, 0.0f, 0.0f), 0.9f, vec3(0.5f, 0.8f, 0.9f), 60.0f, 0.0004f, 270.0f});
    planets.push_back({vec3(70.0f, 0.0f, 0.0f), 0.8f, vec3(0.2f, 0.3f, 0.9f), 70.0f, 0.0001f, 315.0f});

    int planetIndex = 0;
    for (auto &planet : planets) {
        planet.rotationAngle = 0.0f;
        planet.rotationSpeed = 0.1f;
        planetIndex++;
        if (planetIndex <= 2) continue; // skip suns

        if (planet.orbitRadius == 27.0f) {
            planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f), 0.2f, vec3(0.7f, 0.7f, 0.7f), 2.0f, 0.05f, 0.0f});
            planet.rotationSpeed = 0.2f;
            loadSatelliteModels();
        }
        if (planet.orbitRadius == 32.0f) {
            planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f), 0.1f, vec3(0.6f, 0.5f, 0.4f), 1.5f, 0.08f, 0.0f});
            planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f), 0.08f, vec3(0.5f, 0.4f, 0.3f), 2.0f, 0.06f, 90.0f});
        }
        if (planet.orbitRadius == 42.0f) {
            planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f), 0.25f, vec3(0.8f, 0.7f, 0.6f), 3.0f, 0.03f, 0.0f});
            planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f), 0.22f, vec3(0.7f, 0.6f, 0.5f), 4.0f, 0.02f, 90.0f});
            planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f), 0.2f, vec3(0.6f, 0.5f, 0.4f), 5.0f, 0.015f, 180.0f});
            planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f), 0.18f, vec3(0.5f, 0.4f, 0.3f), 6.0f, 0.01f, 270.0f});
            planet.rotationSpeed = 0.3f;
        }
    }
}

void updateSatellites(float deltaTime, const vec3& earthPosition) {
    for (auto& sat : satellites) {
        sat.currentAngle += sat.orbitSpeed * deltaTime * 60.0f;

        float x = sat.orbitRadius * cos(radians(sat.currentAngle));
        float z = sat.orbitRadius * sin(radians(sat.currentAngle)) * cos(radians(sat.tiltAngle));
        float y = sat.orbitRadius * sin(radians(sat.currentAngle)) * sin(radians(sat.tiltAngle));

        sat.position = earthPosition + vec3(x, y, z);
        sat.rotationAngle += sat.rotationSpeed * deltaTime * 60.0f;
    }
}

// planet and twin sun orbits around the center
void updatePlanets(float deltaTime) {
    for (auto &planet : planets) {
        planet.currentAngle += planet.orbitSpeed * deltaTime * 60.0f;
        planet.position.x = planet.orbitRadius * cos(radians(planet.currentAngle));
        planet.position.z = planet.orbitRadius * sin(radians(planet.currentAngle));
        planet.rotationAngle += planet.rotationSpeed * deltaTime * 60.0f;

        for (auto &moon : planet.moons) {
            moon.currentAngle += moon.orbitSpeed * deltaTime * 60.0f;
            moon.offset.x = moon.orbitRadius * cos(radians(moon.currentAngle));
            moon.offset.z = moon.orbitRadius * sin(radians(moon.currentAngle));
        }
    }
}

void compileAndLinkShaders() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

GLuint compileSatelliteShaders() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &satelliteVertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &satelliteFragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

// creating ring for planets
void createRingVAO(float innerRadius, float outerRadius, int segments = 64) {
    std::vector<vec3> vertices;
    std::vector<vec2> texCoords;
    std::vector<unsigned int> indices;

    static bool createdOnce = false;
    if (createdOnce) return; // avoid rebuilding every frame
    createdOnce = true;

    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * glm::pi<float>() * i / segments;
        float x = cos(angle);
        float z = sin(angle);

        vertices.push_back(vec3(outerRadius * x, 0.0f, outerRadius * z));
        texCoords.push_back(vec2(1.0f, i / (float)segments));

        vertices.push_back(vec3(innerRadius * x, 0.0f, innerRadius * z));
        texCoords.push_back(vec2(0.0f, i / (float)segments));
    }

    for (int i = 0; i < segments; i++) {
        indices.push_back(i * 2);
        indices.push_back(i * 2 + 1);
        indices.push_back((i * 2 + 2) % (segments * 2 + 2));

        indices.push_back(i * 2 + 1);
        indices.push_back((i * 2 + 3) % (segments * 2 + 2));
        indices.push_back((i * 2 + 2) % (segments * 2 + 2));
    }

    ringIndexCount = (int)indices.size();

    glGenVertexArrays(1, &ringVAO);
    glBindVertexArray(ringVAO);

    glGenBuffers(1, &ringVBO);
    glBindBuffer(GL_ARRAY_BUFFER, ringVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), &vertices[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &ringEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ringEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    glBindVertexArray(0);
}

// ====== Gun buffers ======
unsigned int gunVAO=0, gunVBO=0, gunEBO=0;
void setupGun() {
    glGenVertexArrays(1, &gunVAO);
    glGenBuffers(1, &gunVBO);
    glGenBuffers(1, &gunEBO);

    glBindVertexArray(gunVAO);

    glBindBuffer(GL_ARRAY_BUFFER, gunVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(gunVertices), gunVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gunEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(gunIndices), gunIndices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

void setupShip() {
    glGenVertexArrays(1, &shipVAO);
    glGenBuffers(1, &shipVBO);
    glGenBuffers(1, &shipEBO);

    glBindVertexArray(shipVAO);
    glBindBuffer(GL_ARRAY_BUFFER, shipVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(shipVertices), shipVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shipEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(shipIndices), shipIndices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

void setupQuad() {
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glGenBuffers(1, &quadEBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIdx), quadIdx, GL_STATIC_DRAW);

    // aPos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // aTexCoords
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
}

// -------- guns & ship rendering --------
inline void getGunTipPositions(glm::vec3& leftTip, glm::vec3& rightTip) {
    vec3 rightV = normalize(cross(cameraFront, cameraUp));
    vec3 upV    = normalize(cameraUp);
    vec3 fwdV   = normalize(cameraFront);

    float t = (float)glfwGetTime();
    float bobY = 0.02f * sin(t * 2.4f);
    float swayX = 0.02f * sin(t * 1.8f + 1.57f);

    vec3 base = cameraPos + fwdV * 1.6f + upV * (-0.55f) + rightV * swayX + upV * bobY;
    leftTip  = base - rightV * 0.32f + fwdV * 0.35f;
    rightTip = base + rightV * 0.32f + fwdV * 0.35f;
}

void renderGuns(const mat4 &view, const mat4 &projection) {
    glUseProgram(shaderProgram);
    glUniform1f(glGetUniformLocation(shaderProgram, "uAlpha"), 1.0f);

    vec3 rightV = normalize(cross(cameraFront, cameraUp));
    vec3 upV    = normalize(cameraUp);
    vec3 fwdV   = normalize(cameraFront);

    mat4 orient = mat4(
        vec4(rightV, 0.0f),
        vec4(upV,    0.0f),
        vec4(fwdV,   0.0f),
        vec4(0,0,0,1)
    );

    float t = static_cast<float>(glfwGetTime());
    float bobY = 0.02f * sin(t * 2.4f);
    float swayX = 0.02f * sin(t * 1.8f + 1.57f);

    vec3 base = cameraPos + fwdV * 1.6f + upV * (-0.55f) + rightV * swayX + upV * bobY;
    vec3 leftPos  = base - rightV * 0.32f;
    vec3 rightPos = base + rightV * 0.32f;

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 0);
    glUniform3f(glGetUniformLocation(shaderProgram, "color"), 0.7f, 0.7f, 0.7f);

    glBindVertexArray(gunVAO);

    mat4 scaleGun = scale(mat4(1.0f), vec3(0.9f, 0.9f, 1.2f));

    // LEFT
    mat4 model = translate(mat4(1.0f), leftPos) * orient * scaleGun;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    // RIGHT
    model = translate(mat4(1.0f), rightPos) * orient * scaleGun;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
}

void renderSpaceshipFirstPerson(const mat4 &view, const mat4 &projection) {
    vec3 rightV = normalize(cross(cameraFront, cameraUp));
    vec3 upV    = normalize(cameraUp);
    vec3 fwdV   = normalize(cameraFront);

    float t = static_cast<float>(glfwGetTime());
    float bobY = 0.02f * sin(t * 2.4f);
    float swayX = 0.02f * sin(t * 1.8f + 1.57f);

    vec3 base = cameraPos + fwdV * 1.4f + upV * (-0.62f) + rightV * swayX + upV * bobY;

    mat4 orient = mat4(
        vec4(rightV, 0.0f),
        vec4(upV,    0.0f),
        vec4(fwdV,   0.0f),
        vec4(0,0,0,1)
    );

    glUseProgram(shaderProgram);
    glUniform1f(glGetUniformLocation(shaderProgram, "uAlpha"), 1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 0);

    GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
    if (depthEnabled) glDisable(GL_DEPTH_TEST);

    glBindVertexArray(shipVAO);

    // Body
    {
        mat4 model = translate(mat4(1.0f), base) * orient *
                     scale(mat4(1.0f), vec3(0.6f, 0.25f, 1.0f));
        glUniform3f(glGetUniformLocation(shaderProgram, "color"), 0.18f, 0.18f, 0.18f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }
    // Canopy
    {
        vec3 canopyOffset = vec3(0.0f, 0.08f, -0.15f);
        mat4 model = translate(mat4(1.0f), base + canopyOffset) * orient *
                     scale(mat4(1.0f), vec3(0.35f, 0.18f, 0.25f));
        glUniform3f(glGetUniformLocation(shaderProgram, "color"), 0.1f, 0.35f, 0.55f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }

    glBindVertexArray(0);

    if (depthEnabled) glEnable(GL_DEPTH_TEST);

    // Guns after ship so they sit in front
    renderGuns(view, projection);
}

// -------- projectiles --------
void spawnBulletPair() {
    vec3 leftTip, rightTip;
    getGunTipPositions(leftTip, rightTip);
    vec3 dir = normalize(cameraFront);

    projectiles.push_back(ProjectileEx(leftTip,  dir * 80.0f,  0.08f, 2.0f, false, false));
    projectiles.push_back(ProjectileEx(rightTip, dir * 80.0f,  0.08f, 2.0f, false, false));

    vec3 rightV = normalize(cross(cameraFront, cameraUp));
    vec3 upV    = normalize(cameraUp);
    flashes.push_back({leftTip,  rightV, upV, 0.09f});
    flashes.push_back({rightTip, rightV, upV, 0.09f});

    // AUDIO: double-barrel
    PlaySfx(SFX_DOUBLE_BARREL, 0.9f);
}

void spawnGrenade() {
    vec3 leftTip, rightTip;
    getGunTipPositions(leftTip, rightTip);
    vec3 dir = normalize(cameraFront);

    vec3 start = (leftTip + rightTip) * 0.5f;
    vec3 vel   = dir * 35.0f + vec3(0.0f, 2.5f, 0.0f);
    projectiles.push_back(ProjectileEx(start, vel, 0.18f, 6.0f, true, false));

    vec3 rightV = normalize(cross(cameraFront, cameraUp));
    vec3 upV    = normalize(cameraUp);
    flashes.push_back({start, rightV, upV, 0.11f});

    // AUDIO: grenade
    PlaySfx(SFX_GRENADE, 0.95f);
}

void spawnLaser() {
    glm::vec3 leftTip, rightTip;
    getGunTipPositions(leftTip, rightTip);
    glm::vec3 origin = (leftTip + rightTip) * 0.5f;
    glm::vec3 dir = normalize(cameraFront);

    projectiles.push_back(ProjectileEx(origin, dir * LASER_SPEED, LASER_RADIUS, LASER_LIFETIME, false, true));

    // a quick flash at the muzzle
    glm::vec3 rightV = normalize(cross(cameraFront, cameraUp));
    glm::vec3 upV    = normalize(cameraUp);
    flashes.push_back({origin, rightV, upV, 0.08f});

    // AUDIO: sniper / laser
    PlaySfx(SFX_SNIPER, 0.8f);
}

const float GRAVITY_Y = -9.8f * 0.6f;
float bulletCooldown = 0.0f;
float grenadeCooldown = 0.0f;

void updateProjectiles(float dt) {
    for (auto& p : projectiles) {
        if (p.grenade) p.vel.y += GRAVITY_Y * dt;
        p.pos += p.vel * dt;
        p.life -= dt;
    }
    projectiles.erase(remove_if(projectiles.begin(), projectiles.end(),
                   [](const ProjectileEx& p){ return p.life <= 0.0f; }),
                   projectiles.end());

    for (auto& f : flashes) f.life -= dt;
    flashes.erase(remove_if(flashes.begin(), flashes.end(),
                   [](const MuzzleFlash& f){ return f.life <= 0.0f; }),
                   flashes.end());
}

void renderProjectiles(const mat4& view, const mat4& proj) {
    glUseProgram(shaderProgram);
    glUniform1f(glGetUniformLocation(shaderProgram, "uAlpha"), 1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &proj[0][0]);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 0);

    glBindVertexArray(sphereVAO);
    for (const auto& p : projectiles) {
        mat4 model = translate(mat4(1.0f), p.pos) * scale(mat4(1.0f), vec3(p.radius));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);

        if (p.grenade) {
            glUniform3f(glGetUniformLocation(shaderProgram, "color"), 0.2f, 0.9f, 0.25f);
        } else if (p.laser) {
            glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 0.1f, 0.1f);
        } else {
            glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 0.95f, 0.2f);
        }

        glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);
}

void renderMuzzleFlashes(const mat4& view, const mat4& proj) {
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &proj[0][0]);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 1); // unlit flashes

    // additive blending for bright bloom
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glBindVertexArray(quadVAO);
    for (const auto& f : flashes) {
        float t = glm::clamp(f.life / 0.11f, 0.0f, 1.0f); // fade
        float sizeX = 0.28f * (0.6f + 0.4f * t);
        float sizeY = 0.28f * (0.6f + 0.4f * t);

        vec3 front = normalize(cross(f.right, f.up));
        mat4 orient = mat4(
            vec4(f.right, 0.0f),
            vec4(f.up,    0.0f),
            vec4(front,   0.0f),
            vec4(0,0,0,1)
        );

        mat4 model = translate(mat4(1.0f), f.pos) * orient * scale(mat4(1.0f), vec3(sizeX, sizeY, 1.0f));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);

        glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 0.6f, 0.15f);
        glUniform1f(glGetUniformLocation(shaderProgram, "uAlpha"), 0.85f * t);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);

    // restore normal blending and alpha
    glUniform1f(glGetUniformLocation(shaderProgram, "uAlpha"), 1.0f);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

// displays stars to gpu
void renderStars(const mat4 &view, const mat4 &projection) {
    glUseProgram(shaderProgram);
    glUniform1f(glGetUniformLocation(shaderProgram, "uAlpha"), 1.0f);

    mat4 model = mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
    glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 1.0f, 1.0f);

    glBindVertexArray(starVAO);
    glDrawArrays(GL_POINTS, 0, (GLsizei)starPositions.size());
    glBindVertexArray(0);
}

// load image textures
unsigned int loadTexture(const char *path) {
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true);

    unsigned char *data = stbi_load(path, &width, &height, &nrChannels, 0);
    if (!data) {
        std::cerr << "Failed to load texture at path: " << path << std::endl;
        return 0;
    }

    GLenum format = GL_RGB;
    if (nrChannels == 1) format = GL_RED;
    else if (nrChannels == 3) format = GL_RGB;
    else if (nrChannels == 4) format = GL_RGBA;

    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);
    return textureID;
}

void setupBuffers() {
    GLfloat *positions = new GLfloat[starPositions.size() * 3];
    GLfloat *sizes = new GLfloat[starSizes.size()];

    int i = 0;
    for (const auto &pos : starPositions) {
        positions[i * 3]     = pos.x;
        positions[i * 3 + 1] = pos.y;
        positions[i * 3 + 2] = pos.z;
        i++;
    }
    i = 0;
    for (const auto &size : starSizes) sizes[i++] = size;

    glGenVertexArrays(1, &starVAO);
    glGenBuffers(1, &starVBO);
    glGenBuffers(1, &starSizeVBO);

    glBindVertexArray(starVAO);

    glBindBuffer(GL_ARRAY_BUFFER, starVBO);
    glBufferData(GL_ARRAY_BUFFER, starPositions.size() * 3 * sizeof(GLfloat), positions, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void *)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, starSizeVBO);
    glBufferData(GL_ARRAY_BUFFER, starSizes.size() * sizeof(GLfloat), sizes, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), (void *)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    delete[] positions;
    delete[] sizes;

    createSphere(1.0f, 36, 18);
}

// ==================== Targets: helpers ====================
glm::vec3 getTargetWorldPos(const TargetBox& t) {
    int idx = 0;
    for (const auto &p : planets) {
        if (idx == t.planetIndex) {
            return p.position + t.offset;
        }
        idx++;
    }
    return t.offset;
}

// Call once after initSolarSystem()
void generateTargets(int perPlanet = 5, float minR=2.5f, float maxR=6.5f) {
    targets.clear();
    int pIndex = 0;
    for (const auto &p : planets) {
        if (pIndex < 2) { pIndex++; continue; } // skip suns
        for (int i=0;i<perPlanet;i++) {
            float r = minR + (maxR-minR) * (rand()/(float)RAND_MAX);
            float theta = 2.0f * glm::pi<float>() * (rand()/(float)RAND_MAX);
            float phi = (glm::pi<float>() * 0.5f) * ((rand()/(float)RAND_MAX) - 0.5f);
            glm::vec3 off(
                r * cosf(theta) * cosf(phi),
                r * sinf(phi),
                r * sinf(theta) * cosf(phi)
            );
            TargetBox t;
            t.planetIndex = pIndex;
            t.offset = off;
            t.size = 0.35f;   // half-size
            t.lasting
     = true;
            targets.push_back(t);
        }
        pIndex++;
    }
}

void setupTargetCube() {
    static float cubeVerts[] = {
        -1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1,
        -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1
    };
    static unsigned int cubeIdx[] = {
        0,1,2, 2,3,0,  4,5,6, 6,7,4,
        0,3,7, 7,4,0,  1,5,6, 6,2,1,
        3,2,6, 6,7,3,  0,4,5, 5,1,0
    };
    glGenVertexArrays(1,&targetVAO);
    glGenBuffers(1,&targetVBO);
    glGenBuffers(1,&targetEBO);
    glBindVertexArray(targetVAO);
    glBindBuffer(GL_ARRAY_BUFFER, targetVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVerts), cubeVerts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, targetEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIdx), cubeIdx, GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

void renderTargets(const glm::mat4& view, const glm::mat4& proj) {
    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 0);
    glUniform1f(glGetUniformLocation(shaderProgram, "uAlpha"), 1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &proj[0][0]);

    glBindVertexArray(targetVAO);
    for (const auto& t : targets) {
        if (!t.lasting
) continue;
        glm::vec3 wp = getTargetWorldPos(t);
        glm::mat4 model(1.0f);
        model = glm::translate(model, wp);
        model = glm::scale(model, glm::vec3(t.size)); // half-size cube
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        // Make targets teal-ish
        glUniform3f(glGetUniformLocation(shaderProgram, "color"), 0.1f, 0.9f, 0.8f);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);
}

void checkProjectileTargetCollisions() {
    for (auto& p : projectiles) {
        if (p.life <= 0.0f) continue;
        for (auto& t : targets) {
            if (!t.lasting
    ) continue;
            glm::vec3 tw = getTargetWorldPos(t);
            // sphere vs AABB (centered): use distance from box
            glm::vec3 d = glm::abs(p.pos - tw) - glm::vec3(t.size);
            float distOutside = glm::length(glm::max(d, glm::vec3(0.0f))) + std::min(std::max(d.x,std::max(d.y,d.z)), 0.0f);
            if (distOutside <= p.radius) {
                t.lasting
         = false;
                p.life = 0.0f;
                gScore++;
                // small flash at impact
                glm::vec3 rightV = normalize(cross(cameraFront, cameraUp));
                glm::vec3 upV    = normalize(cameraUp);
                flashes.push_back({tw, rightV, upV, 0.12f});
                break;
            }
        }
    }
}

// ==================== Scope overlay setup & draw ====================
void setupScopeOverlay() {
    // Fullscreen quad in NDC with UV
    float verts[] = {
        -1.f, -1.f,  0.f, 0.f,
         1.f, -1.f,  1.f, 0.f,
         1.f,  1.f,  1.f, 1.f,
        -1.f,  1.f,  0.f, 1.f
    };
    unsigned int idx[] = {0,1,2, 2,3,0};

    glGenVertexArrays(1,&scopeVAO);
    glGenBuffers(1,&scopeVBO);
    glGenBuffers(1,&scopeEBO);

    glBindVertexArray(scopeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, scopeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scopeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // Compile program
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs,1,&scopeVS,nullptr);
    glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs,1,&scopeFS,nullptr);
    glCompileShader(fs);
    scopeProg = glCreateProgram();
    glAttachShader(scopeProg, vs);
    glAttachShader(scopeProg, fs);
    glLinkProgram(scopeProg);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

void renderScopeOverlay(int winW, int winH) {
    if (!gScoped) return;

    GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
    if (depthEnabled) glDisable(GL_DEPTH_TEST);
    glUseProgram(scopeProg);
    glBindVertexArray(scopeVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUniform1f(glGetUniformLocation(scopeProg, "aspect"), (float)winW / (float)winH);
    glUniform1f(glGetUniformLocation(scopeProg, "alpha"), 0.85f); // darkness outside
    glUniform1i(glGetUniformLocation(scopeProg, "drawCross"), 1);
    glUniform1f(glGetUniformLocation(scopeProg, "lineW"), 0.006f);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
    if (depthEnabled) glEnable(GL_DEPTH_TEST);
}

int main() {
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    stbi_set_flip_vertically_on_load(true);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Solar System", NULL, NULL);
    if (!window) {
        cerr << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (glewInit() != GLEW_OK) {
        cerr << "Failed to initialize GLEW" << endl;
        return -1;
    }

    // ---------------- AUDIO INIT ----------------
    {
        ma_result r = ma_engine_init(NULL, &gAudioEngine);
        if (r != MA_SUCCESS) {
            std::cerr << "Failed to initialize audio engine (miniaudio)." << std::endl;
            // continue without audio
        }
    }
    // -------------------------------------------

    // load planet textures
    unsigned int sunTexture     = loadTexture("Textures/Sun.png");
    unsigned int mercuryTexture = loadTexture("Textures/Mercury.jpg");
    unsigned int venusTexture   = loadTexture("Textures/Venus.jpg");
    unsigned int earthTexture   = loadTexture("Textures/Earth.jpg");
    unsigned int marsTexture    = loadTexture("Textures/Mars.jpg");
    unsigned int jupiterTexture = loadTexture("Textures/Jupiter.jpg");
    unsigned int saturnTexture  = loadTexture("Textures/Saturn.jpg");
    unsigned int uranusTexture  = loadTexture("Textures/Uranus.jpg");
    unsigned int neptuneTexture = loadTexture("Textures/Neptune.jpg");

    std::vector<GLuint> planetTextures = {
        sunTexture, sunTexture, mercuryTexture, venusTexture, earthTexture,
        marsTexture, jupiterTexture, saturnTexture, uranusTexture, neptuneTexture
    };

    // load moon textures
    unsigned int moonTexture    = loadTexture("Textures/The moon.jpg");
    unsigned int phobosTexture  = loadTexture("Textures/Phobos.jpg");
    unsigned int deimosTexture  = loadTexture("Textures/Deimos.jpg");
    unsigned int ioTexture      = loadTexture("Textures/Io.jpg");
    unsigned int europaTexture  = loadTexture("Textures/Europa.jpg");
    unsigned int ganymedeTexture= loadTexture("Textures/Ganymede.jpg");
    unsigned int callistoTexture= loadTexture("Textures/Callisto.jpg");

    std::vector<GLuint> moonTextures = {
        moonTexture, phobosTexture, deimosTexture, ioTexture, europaTexture, ganymedeTexture, callistoTexture
    };

    srand(static_cast<unsigned int>(time(nullptr)));
    generateStarField();
    initSolarSystem();
    compileAndLinkShaders();
    GLuint satProg = compileSatelliteShaders();
    loadSatelliteModels();
    setupBuffers();
    setupGun();
    setupShip();
    setupQuad();
    setupTargetCube();
    setupScopeOverlay();
    generateTargets(6); // e.g., 6 boxes per non-sun planet
    createRingVAO(0.9f, 1.3f);

    // make asteroids
    loadAsteroidModels();
    generateAsteroidBelt(300, 35.0f, 40.0f);

    // opengl states
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    float titleTimer = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        bulletCooldown  = std::max(0.0f, bulletCooldown - deltaTime);
        grenadeCooldown = std::max(0.0f, grenadeCooldown - deltaTime);
        laserCooldown   = std::max(0.0f, laserCooldown - deltaTime);

        // input + movement
        processInput(window, deltaTime);

        // firing
        if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS && bulletCooldown <= 0.0f) {
            spawnBulletPair();
            bulletCooldown = 0.12f; // ~8.3/s
        }
        if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS && grenadeCooldown <= 0.0f) {
            spawnGrenade();
            grenadeCooldown = 0.6f;
        }

        // Scope toggle on C (edge-trigger)
        bool cNow = (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS);
        if (cNow && !gScopeKeyHeld) {
            gScoped = !gScoped;
        }
        gScopeKeyHeld = cNow;

        // Laser on L
        bool lNow = (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS);
        if (lNow && laserCooldown <= 0.0f) {
            spawnLaser();
            laserCooldown = LASER_COOLDOWN;
        }

        updatePlanets(deltaTime);
        updateAsteroids(deltaTime); // hassan    keeps it spinning and moving
        updateProjectiles(deltaTime);

        // smooth FOV toward target
        float targetFov = gScoped ? FOV_SCOPED : FOV_NORMAL;
        gCurrentFov += (targetFov - gCurrentFov) * (1.0f - expf(-FOV_SMOOTH * deltaTime));

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mat4 view = lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        mat4 projection = perspective(radians(gCurrentFov),
                                      (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT,
                                      0.1f,
                                      MAX_DISTANCE * 2.0f);

        // satellites
        vec3 earthPos;
        int index = 0;
        for (const auto &planet : planets) {
            if (index == 4) { earthPos = planet.position; break; }
            index++;
        }
        updateSatellites(deltaTime, earthPos);
        renderSatellites(view, projection, satProg);

        // stars
        glUseProgram(shaderProgram);
        glUniform1f(glGetUniformLocation(shaderProgram, "uAlpha"), 1.0f);
        glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
        renderStars(view, projection);
        renderAsteroids(view, projection, shaderProgram);

        // planets & suns
        glUseProgram(shaderProgram);
        glUniform1f(glGetUniformLocation(shaderProgram, "uAlpha"), 1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);

        vec3 sunPositions[2];
        vec3 sunColors[2];
        int sunIdx = 0;
        for (auto it = planets.begin(); it != planets.end() && sunIdx < 2; ++it, ++sunIdx) {
            sunPositions[sunIdx] = it->position;
            sunColors[sunIdx] = it->color;
        }
        glUniform3fv(glGetUniformLocation(shaderProgram, "sunPositions"), 2, &sunPositions[0][0]);
        glUniform3fv(glGetUniformLocation(shaderProgram, "sunColors"), 2, &sunColors[0][0]);
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, &cameraPos[0]);

        int i = 0;
        int moonTextureIndex = 0;
        for (auto it = planets.begin(); it != planets.end(); ++it, i++) {
            Planet &planet = *it;
            mat4 model = mat4(1.0f);
            model = translate(model, planet.position);
            model = rotate(model, radians(planet.rotationAngle), vec3(0.0f, 1.0f, 0.0f));
            model = scale(model, vec3(planet.radius));

            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);

            int isSun = (i < 2) ? 1 : 0;
            glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), isSun);

            if (i < (int)planetTextures.size()) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, planetTextures[i]);
                glUniform1i(glGetUniformLocation(shaderProgram, "planetTexture"), 0);
                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);
            } else {
                glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &planet.color[0]);
                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
            }

            if (planet.hasRing) {
                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
                glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &planet.ringColor[0]);
                glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 0);

                mat4 ringModel = mat4(1.0f);
                ringModel = translate(ringModel, planet.position);
                ringModel = rotate(ringModel, radians(planet.rotationAngle * 0.5f), vec3(0.0f, 1.0f, 0.0f));
                ringModel = rotate(ringModel, radians(25.0f), vec3(1.0f, 0.0f, 0.0f));
                float ringScale = planet.radius * 1.5f;
                ringModel = scale(ringModel, vec3(ringScale));

                glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &ringModel[0][0]);
                glBindVertexArray(ringVAO);
                glDrawElements(GL_TRIANGLES, ringIndexCount, GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);

                glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);
            }

            glBindVertexArray(sphereVAO);
            glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);

            for (const auto &moon : planet.moons) {
                mat4 moonModel = mat4(1.0f);
                moonModel = translate(moonModel, planet.position + moon.offset);
                moonModel = scale(moonModel, vec3(moon.radius));

                glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &moonModel[0][0]);
                glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 0);

                if (moonTextureIndex < (int)moonTextures.size()) {
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, moonTextures[moonTextureIndex]);
                    glUniform1i(glGetUniformLocation(shaderProgram, "planetTexture"), 0);
                    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);
                } else {
                    glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &moon.color[0]);
                    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
                }

                glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);
                moonTextureIndex++;
            }

            glBindVertexArray(0);
        }

        // Draw targets in-world
        renderTargets(view, projection);

        // in-world projectiles (after targets so they can overlap)
        renderProjectiles(view, projection);

        // overlay: ship + guns, then flashes on top
        renderSpaceshipFirstPerson(view, projection);
        renderMuzzleFlashes(view, projection);

        // collisions after projectile move
        checkProjectileTargetCollisions();

        // scope overlay last
        renderScopeOverlay(WINDOW_WIDTH, WINDOW_HEIGHT);

        // Update window title with score occasionally
        titleTimer += deltaTime;
        if (titleTimer > 0.25f) {
            titleTimer = 0.0f;
            std::string title = "Solar System - Score: " + std::to_string(gScore) + (gScoped ? "  [SCOPED]" : "");
            glfwSetWindowTitle(window, title.c_str());
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // clean up
    glDeleteVertexArrays(1, &sphereVAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &sphereEBO);
    glDeleteVertexArrays(1, &starVAO);
    glDeleteBuffers(1, &starVBO);
    glDeleteBuffers(1, &starSizeVBO);
    glDeleteVertexArrays(1, &gunVAO);
    glDeleteBuffers(1, &gunVBO);
    glDeleteBuffers(1, &gunEBO);
    glDeleteVertexArrays(1, &shipVAO);
    glDeleteBuffers(1, &shipVBO);
    glDeleteBuffers(1, &shipEBO);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteBuffers(1, &quadEBO);
    glDeleteVertexArrays(1, &targetVAO);
    glDeleteBuffers(1, &targetVBO);
    glDeleteBuffers(1, &targetEBO);
    glDeleteVertexArrays(1, &ringVAO);
    glDeleteBuffers(1, &ringVBO);
    glDeleteBuffers(1, &ringEBO);

    glDeleteProgram(scopeProg);
    glDeleteProgram(shaderProgram);

    // ------------- AUDIO SHUTDOWN -------------
    ma_engine_uninit(&gAudioEngine);
    // ------------------------------------------

    glfwTerminate();
    return 0;
}
// Bibliogrpahy:
//  Note that many parts were done using google and chatgpt aswell for learning and implmentation.
