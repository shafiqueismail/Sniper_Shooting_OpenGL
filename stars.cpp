    // creates a "star box" centered around the origin
    // wasd + up/down movement available, along with a speedup
    // movement is based on current camera direction

    #include <iostream>
    #include <list>
    #include <vector>

    #define GLEW_STATIC 1                   // This allows linking with Static Library on Windows, without DLL
    #include <GL/glew.h>                    // Include GLEW - OpenGL Extension Wrangler
    #include <GLFW/glfw3.h>                 // cross-platform interface for creating a graphical context, initializing OpenGL and binding inputs
    #include <glm/glm.hpp>                  // GLM is an optimized math library with syntax to similar to OpenGL Shading Language
    #include <glm/gtc/matrix_transform.hpp> // include this to create transformation matrices
    #include <glm/common.hpp>
    #define STB_IMAGE_IMPLEMENTATION
    #include <stb/stb_image.h> // image loading library

    #include <glm/gtx/transform.hpp>
    #include "OBJloader.h" 

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

    // moons
    struct Moon
    {
        vec3 offset;        // current position relative to planet (updated each frame)
        float radius;       // size of the moon
        vec3 color;         // RGB color of the moon
        float orbitRadius;  // distance from the planet
        float orbitSpeed;   // speed of moon orbits (degrees/frame)
        float currentAngle; // current position in orbit (0-360 degrees)
    };

    // solar system parameters
    struct Planet
    {
        vec3 position;
        float radius;
        vec3 color;
        float orbitRadius;
        float orbitSpeed;
        float currentAngle;
        float rotationAngle; // planet's rotation
        float rotationSpeed; // rotation speed
        list<Moon> moons;    // moons orbiting the planet

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

        void setupBuffers()
        {
            glGenVertexArrays(1, &vao);
            glGenBuffers(2, vbo);

            glBindVertexArray(vao);

            // Vertices
            glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

            // Normals
            if (!normals.empty())
            {
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
            // Approximate normal for sphere: just use normalized position
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

        in vec2 TexCoord;
        in vec3 FragPos;
        in vec3 Normal;
        out vec4 FragColor;

        void main()
        {
            // get base color 
            vec4 baseColor = (useTexture == 1) ? texture(planetTexture, TexCoord) : vec4(color, 1.0);

            // prevents unnecessary calculations
            if (isSun == 1) {
                FragColor = vec4(baseColor.rgb, 1.0);
                return;
            }

            // calculate light using normal vector
            vec3 norm = normalize(Normal);
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 ambient = 0.15 * baseColor.rgb;
            vec3 lighting = ambient;

            // process each light source
            for (int i = 0; i < 2; ++i) {
                // calculate light direction
                vec3 lightDir = normalize(sunPositions[i] - FragPos);
                
                // diffuse component (Lambert)
                float diff = max(dot(norm, lightDir), 0.0);
                float spec = 0.0;

                // specular component (Phong)
                if (diff > 0.0) {
                    vec3 reflectDir = reflect(-lightDir, norm);
                    spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
                }

                // calculations
                vec3 diffuse = diff * sunColors[i] * baseColor.rgb;
                vec3 specular = 0.3 * spec * sunColors[i];

                // combine lighting
                lighting += diffuse + specular;
            }
            FragColor = vec4(lighting, baseColor.a);
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
                vec3 specular = 0.3 * spec * sunColors[i]; /// 0.3 * spec
                
                lighting += (diffuse + specular);
            }
            
            FragColor = vec4(lighting * vec3(0.1411764706,0.1411764706,0.1411764706), 1.0); // 0.1411764706 dark grey color satellite
        }
    )glsl";

    GLuint shaderProgram;
    GLuint sphereVAO, sphereVBO;          // planet array/buffer
    GLuint starVAO, starVBO, starSizeVBO; // star array/buffer
    list<vec3> starPositions;
    list<float> starSizes;

    // mouse movement
    void mouse_callback(GLFWwindow *window, double xpos, double ypos)
    {
        if (firstMouse)
        {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;

        float sensitivity = 0.005f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        vec3 front;
        front.x = cos(radians(yaw)) * cos(radians(pitch));
        front.y = sin(radians(pitch));
        front.z = sin(radians(yaw)) * cos(radians(pitch));
        cameraFront = normalize(front);
    }

    // for keyboard movement
    void processInput(GLFWwindow *window, float deltaTime)
    {
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
    void generateStarField()
    {
        // for all stars
        for (int i = 0; i < NUM_STARS; i++)
        {
            // distance and angle
            float distance = MAX_DISTANCE * pow(static_cast<float>(rand()) / RAND_MAX, 0.25f);
            float theta = 2.0f * pi<float>() * static_cast<float>(rand()) / RAND_MAX;
            float phi = acos(2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);

            // position calculation
            float x = distance * sin(phi) * cos(theta);
            float y = distance * sin(phi) * sin(theta);
            float z = distance * cos(phi);

            // adds position vector to list
            starPositions.push_back(vec3(x, y, z));

            // size generation (smaller is more likely than large)
            float size = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * pow(static_cast<float>(rand()) / RAND_MAX, 4.0f);
            starSizes.push_back(size);
        }
    }

    GLuint sphereEBO;              // element buffer object for sphere indices
    unsigned int sphereIndexCount; // indices for the sphere

    // generates sphere for planets, suns, moons
    void createSphere(float radius, int sectors, int stacks)
    {
        // memory allocation
        int vertexCount = (stacks + 1) * (sectors + 1);
        int indexCount = stacks * sectors * 6;        // 2 triangles per sector, 3 indices each
        float *vertices = new float[vertexCount * 5]; // 3 for pos + 2 for UV
        unsigned int *indices = new unsigned int[indexCount];

        // vertex generation
        float sectorStep = 2 * glm::pi<float>() / sectors; // longitude
        float stackStep = glm::pi<float>() / stacks;       // latitude

        int vertexIndex = 0;
        for (int i = 0; i <= stacks; ++i) {  // latitude
            // position calculation
            float stackAngle = glm::pi<float>() / 2 - i * stackStep; // from pi/2 to -pi/2
            float xy = radius * cosf(stackAngle);                    // horizontal radius at current height
            float z = radius * sinf(stackAngle);                     // vertical position

            for (int j = 0; j <= sectors; ++j) { // longitude
                // position calculation
                float sectorAngle = j * sectorStep;
                float x = xy * cosf(sectorAngle);
                float y = xy * sinf(sectorAngle);

                // position
                vertices[vertexIndex++] = x;
                vertices[vertexIndex++] = y;
                vertices[vertexIndex++] = z;

                // texture coordinates (u,v)
                vertices[vertexIndex++] = (float)j / sectors; // u - needed to rotate texture 1.0f - 
                vertices[vertexIndex++] = (float)i / stacks;  // v

            }
        }

        // index buffer generation
        int index = 0;
        for (int i = 0; i < stacks; ++i)
        {
            int k1 = i * (sectors + 1); // current row
            int k2 = k1 + sectors + 1;  // next row

            for (int j = 0; j < sectors; ++j, ++k1, ++k2)
            {
                if (i != 0) {
                    // first triangle (k1 → k2 → k1+1)
                    indices[index++] = k1;
                    indices[index++] = k2;
                    indices[index++] = k1 + 1;
                }

                if (i != (stacks - 1)) {
                    // second triangle (k1+1 → k2 → k2+1)
                    indices[index++] = k1 + 1;
                    indices[index++] = k2;
                    indices[index++] = k2 + 1;
                }
            }
        }

        sphereIndexCount = index; // actual index count

        // generate VAO, VBO, EBO
        glGenVertexArrays(1, &sphereVAO);
        glGenBuffers(1, &sphereVBO);
        glGenBuffers(1, &sphereEBO);

        glBindVertexArray(sphereVAO);

        // vbo
        glBindBuffer(GL_ARRAY_BUFFER, sphereVBO); 
        glBufferData(GL_ARRAY_BUFFER, vertexCount * 5 * sizeof(float), vertices, GL_STATIC_DRAW); // upload to gpu

        // ebo
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndexCount * sizeof(unsigned int), indices, GL_STATIC_DRAW); // upload to gpu

        // position attribute (location = 0) in shader
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0); // location 0 in shader

        // texture coord attribute (location = 2) in shader
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(2); // location 2 in shader

        // unbinds vbo and vao
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        // cleanup
        delete[] vertices;
        delete[] indices;
    }

    void loadSatelliteModels() {
        Satellite tess, quikscat;
        
        std::vector<glm::vec2> temp_uvs;

        // Load TESS model
        if (!loadOBJ("models/TESS.obj", tess.model.vertices, tess.model.normals, temp_uvs)) {
            std::cerr << "Failed to load TESS model" << std::endl;
        }
        tess.model.setupBuffers();
        tess.orbitRadius = 1.0f;
        tess.orbitSpeed = 0.1f;
        tess.currentAngle = 0.0f;
        tess.tiltAngle = 30.0f;
        tess.size = 0.005f;
        tess.rotationAngle = 0.5f;       // Initialize rotation
        tess.rotationSpeed = 0.025f;      // Rotation speed (degrees per frame)

        temp_uvs.clear();

        // Load QuikSCAT model
        if (!loadOBJ("models/QuikSCAT.obj", quikscat.model.vertices, quikscat.model.normals, temp_uvs)) {
            std::cerr << "Failed to load QuikSCAT model" << std::endl;
        }
        quikscat.model.setupBuffers();
        quikscat.orbitRadius = 1.0f;
        quikscat.orbitSpeed = 0.1f;
        quikscat.currentAngle = 180.0f;
        quikscat.tiltAngle = -15.0f;
        quikscat.size = 0.025f;
        quikscat.rotationAngle = -180.0f;  // Initialize rotation
        quikscat.rotationSpeed = 0.075f;  // Slightly different rotation speed

        temp_uvs.clear();

        satellites.push_back(tess);
        satellites.push_back(quikscat);
    }


    void renderSatellites(const mat4 &view, const mat4 &projection, GLuint satelliteShader) {
        glUseProgram(satelliteShader);

        // Find Earth's position
        vec3 earthPos;
        int planetIndex = 0;
        for (const auto &planet : planets)
        {
            if (planetIndex == 3)
            { // Earth is 4th planet (index 3)
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
        for (const auto &planet : planets)
        {
            if (sunCount >= 2)
                break;
            if (planetIndex < 2)
            { // First two planets are suns
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
            
            // Face toward Earth (create look-at matrix)
            vec3 toEarth = normalize(earthPos - sat.position);
            vec3 up = vec3(0.0f, 1.0f, 0.0f);
            vec3 right = normalize(cross(toEarth, up));
            up = normalize(cross(right, toEarth));
            mat4 faceEarth = mat4(vec4(right, 0), vec4(up, 0), vec4(-toEarth, 0), vec4(0, 0, 0, 1));
            
            // Apply horizontal rotation (around Y-axis after facing Earth)
            mat4 rotation = rotate(-radians(sat.rotationAngle), vec3(0, 1, 0));
            
            // Combine transformations: position → face Earth → rotate → scale
            model = model * faceEarth * rotation;
            model = scale(model, vec3(sat.size));

            // Set shader uniforms
            glUniformMatrix4fv(glGetUniformLocation(satelliteShader, "model"), 1, GL_FALSE, &model[0][0]);
            glUniformMatrix4fv(glGetUniformLocation(satelliteShader, "view"), 1, GL_FALSE, &view[0][0]);
            glUniformMatrix4fv(glGetUniformLocation(satelliteShader, "projection"), 1, GL_FALSE, &projection[0][0]);
            
            // Draw satellite
            sat.model.Draw();
        }
    }

    // initial twin star system positions
    void initSolarSystem()
    {
        float sunOrbitRadius = 12.0f; // distance from center for each sun
        float sun1Angle = 0.0f;
        float sun2Angle = 180.0f;    // opposite phase
        float sunOrbitSpeed = 0.18f; // speed of sun orbit

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
            false // no ring
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
            false // no ring
        });

        // All other planets orbit the center
        // mercury
        planets.push_back({vec3(18.0f, 0.0f, 0.0f),
                        0.4f,
                        vec3(0.7f, 0.7f, 0.7f),
                        18.0f,
                        0.04f,
                        0.0f});
        // venus
        planets.push_back({vec3(22.0f, 0.0f, 0.0f),
                        0.6f,
                        vec3(0.9f, 0.6f, 0.2f),
                        22.0f,
                        0.015f,
                        45.0f});
        // earth
        planets.push_back({vec3(27.0f, 0.0f, 0.0f),
                        0.7f,
                        vec3(0.2f, 0.4f, 0.9f),
                        27.0f, //
                        0.01f,
                        90.0f});
        // mars
        planets.push_back({vec3(32.0f, 0.0f, 0.0f),
                        0.5f,
                        vec3(0.8f, 0.3f, 0.1f),
                        32.0f,
                        0.008f,
                        135.0f});
        // jupiter
        planets.push_back({
                        vec3(30.0f, 0.0f, 0.0f),
                        1.5f,
                        vec3(0.8f, 0.6f, 0.4f),
                        42.0f,
                        0.002f,
                        180.0f});
        // saturn
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
            true,                  // has a ring
            0.9f,                  // ringInnerRadius
            1.3f,                  // ringOuterRadius
            vec3(0.6f, 0.7f, 0.9f) // ringColor
        });
        // uranus
        planets.push_back({vec3(60.0f, 0.0f, 0.0f),
                        0.9f,
                        vec3(0.5f, 0.8f, 0.9f),
                        60.0f,
                        0.0004f,
                        270.0f});
        // neptune
        planets.push_back({vec3(70.0f, 0.0f, 0.0f),
                        0.8f,
                        vec3(0.2f, 0.3f, 0.9f),
                        70.0f,
                        0.0001f,
                        315.0f});

        // add moons to planets
        // push_back adds it to the planets' moon list
        int planetIndex = 0;
        for (auto &planet : planets)
        {
            planet.rotationAngle = 0.0f;
            planet.rotationSpeed = 0.1f; // base rotation speed
            planetIndex++;

            if (planetIndex <= 2)
                continue; // skip the two suns

            // earth's moon
            if (planet.orbitRadius == 27.0f)
            {
                planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f),
                                        0.2f,
                                        vec3(0.7f, 0.7f, 0.7f),
                                        2.0f,
                                        0.05f,
                                        0.0f});
                planet.rotationSpeed = 0.2f;
                loadSatelliteModels(); // load satellites with moons
            }

            // mars' moons
            if (planet.orbitRadius == 32.0f)
            {
                planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f),
                                        0.1f,
                                        vec3(0.6f, 0.5f, 0.4f),
                                        1.5f,
                                        0.08f,
                                        0.0f});
                planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f),
                                        0.08f,
                                        vec3(0.5f, 0.4f, 0.3f),
                                        2.0f,
                                        0.06f,
                                        90.0f});
            }

            // jupiter's moons (4 largest)
            if (planet.orbitRadius == 42.0f)
            {
                planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f),
                                        0.25f,
                                        vec3(0.8f, 0.7f, 0.6f),
                                        3.0f,
                                        0.03f,
                                        0.0f});
                planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f),
                                        0.22f,
                                        vec3(0.7f, 0.6f, 0.5f),
                                        4.0f,
                                        0.02f,
                                        90.0f});
                planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f),
                                        0.2f,
                                        vec3(0.6f, 0.5f, 0.4f),
                                        5.0f,
                                        0.015f,
                                        180.0f});
                planet.moons.push_back({vec3(0.0f, 0.0f, 0.0f),
                                        0.18f,
                                        vec3(0.5f, 0.4f, 0.3f),
                                        6.0f,
                                        0.01f,
                                        270.0f});
                planet.rotationSpeed = 0.3f;
            }
        }
    }

    void updateSatellites(float deltaTime, const vec3& earthPosition) {
        for (auto& sat : satellites) {
            // Update orbital position
            sat.currentAngle += sat.orbitSpeed * deltaTime * 60.0f;
            
            // Calculate position with tilt
            float x = sat.orbitRadius * cos(radians(sat.currentAngle));
            float z = sat.orbitRadius * sin(radians(sat.currentAngle)) * cos(radians(sat.tiltAngle));
            float y = sat.orbitRadius * sin(radians(sat.currentAngle)) * sin(radians(sat.tiltAngle));
            
            sat.position = earthPosition + vec3(x, y, z);
            
            // Update horizontal rotation (around Y-axis)
            sat.rotationAngle += sat.rotationSpeed * deltaTime * 60.0f;
        }
    }



    // planet and twin sun orbits around the center
    void updatePlanets(float deltaTime)
    {
        int planetIndex = 0;
        for (auto &planet : planets)
        {
            // calculate position using angle (following orbit)
            // same calculation for simplicity
            planet.currentAngle += planet.orbitSpeed * deltaTime * 60.0f;   
            planet.position.x = planet.orbitRadius * cos(radians(planet.currentAngle));
            planet.position.z = planet.orbitRadius * sin(radians(planet.currentAngle));

            // planet.currentAngle: tracks the planet's current position in its orbit (in degrees)
            // planet.orbitSpeed: degrees per frame (when running at 60 FPS)
            // deltaTime: time since last frame (in seconds)
            // * 60.0f: converts from "degrees per 60Hz frame" to "degrees per second"

            // update planet's rotation
            planet.rotationAngle += planet.rotationSpeed * deltaTime * 60.0f;
            
            // update moons' orbits
            for (auto &moon : planet.moons)
            {
                moon.currentAngle += moon.orbitSpeed * deltaTime * 60.0f;
                moon.offset.x = moon.orbitRadius * cos(radians(moon.currentAngle));
                moon.offset.z = moon.orbitRadius * sin(radians(moon.currentAngle));
            }

            planetIndex++;
        }
    }

    void compileAndLinkShaders()
    {
        // vertex shader
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);

        // fragment shader
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);

        // shader program
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        // cleanup
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }

    GLuint compileSatelliteShaders() {
        // Vertex shader
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &satelliteVertexShaderSource, NULL);
        glCompileShader(vertexShader);

        // Fragment shader
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &satelliteFragmentShaderSource, NULL);
        glCompileShader(fragmentShader);

        // Shader program
        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        // Cleanup
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        return program;
    }

    // creating ring for planets
    unsigned int ringVAO, ringVBO, ringEBO;
    int ringIndexCount;

    void createRingVAO(float innerRadius, float outerRadius, int segments = 64)
    {
        // setup
        std::vector<vec3> vertices;
        std::vector<vec2> texCoords;
        std::vector<unsigned int> indices;

        for (int i = 0; i <= segments; ++i)
        {
            float angle = 2.0f * glm::pi<float>() * i / segments;
            float x = cos(angle);
            float z = sin(angle);

            // outer vertex (outer edge of ring)
            vertices.push_back(vec3(outerRadius * x, 0.0f, outerRadius * z));
            texCoords.push_back(vec2(1.0f, i / (float)segments));

            // inner vertex (inner edge of ring)
            vertices.push_back(vec3(innerRadius * x, 0.0f, innerRadius * z));
            texCoords.push_back(vec2(0.0f, i / (float)segments));
        }

        // create triangle strip indices
        for (int i = 0; i < segments; i++) {
            indices.push_back(i * 2);                            // current outer vertex
            indices.push_back(i * 2 + 1);                        // current inner vertex
            indices.push_back((i * 2 + 2) % (segments * 2 + 2)); // next outer vertex

            indices.push_back(i * 2 + 1);                        // current inner vertex
            indices.push_back((i * 2 + 3) % (segments * 2 + 2)); // next inner vertex
            indices.push_back((i * 2 + 2) % (segments * 2 + 2)); // next outer vertex
        }

        ringIndexCount = indices.size();

        // create and bind VAO (stores all buffer configurations (VBO + EBO) for easy rendering)
        glGenVertexArrays(1, &ringVAO);
        glBindVertexArray(ringVAO);

        // position VBO
        glGenBuffers(1, &ringVBO);              // create vbo
        glBindBuffer(GL_ARRAY_BUFFER, ringVBO); // binds ringVBO to GL_ARRAY_BUFFER (all buffer operations on it affect ringVBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), &vertices[0], GL_STATIC_DRAW); // uploads to gpu
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0); // location 0
        glEnableVertexAttribArray(0);                                 // location 0

        // EBO (stores triangle indices for efficient rendering)
        glGenBuffers(1, &ringEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ringEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // unbind vao
        glBindVertexArray(0);
    }

    void setupBuffers() {
        // star buffers
        GLfloat *positions = new GLfloat[starPositions.size() * 3]; // 3 components (x,y,z) per star
        GLfloat *sizes = new GLfloat[starSizes.size()];             // 1 size value per star

        int i = 0;
        for (const auto &pos : starPositions)
        {
            positions[i * 3] = pos.x;     // copies star positions from starPositions list (which contains vec3) into a contiguous float array
            positions[i * 3 + 1] = pos.y; // layout becomes: [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]
            positions[i * 3 + 2] = pos.z;
            i++;
        }

        i = 0;
        for (const auto &size : starSizes) {  // Copies star sizes into a contiguous float array
            sizes[i++] = size;                // Layout: [size0, size1, size2, ...]
        }

        glGenVertexArrays(1, &starVAO);       // generates Vertex Array Object (VAO) to store buffer configurations
        glGenBuffers(1, &starVBO);            // vbo for star positions
        glGenBuffers(1, &starSizeVBO);        // vbo for star sizes

        // binds the VAO to store configuration
        glBindVertexArray(starVAO);

        // position VBO
        glBindBuffer(GL_ARRAY_BUFFER, starVBO);
        glBufferData(GL_ARRAY_BUFFER, starPositions.size() * 3 * sizeof(GLfloat), positions, GL_STATIC_DRAW); // uploads data to GPU
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void *)0); // sets vertex attribute pointer (using location 0)
        glEnableVertexAttribArray(0); // use the data from attribute (location 0) when rendering

        // size VBO
        glBindBuffer(GL_ARRAY_BUFFER, starSizeVBO);
        glBufferData(GL_ARRAY_BUFFER, starSizes.size() * sizeof(GLfloat), sizes, GL_STATIC_DRAW); // uploads size data to GPU
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), (void *)0); // sets vertex attribute pointer (using location 1)
        glEnableVertexAttribArray(1); // use the data from attribute (location 1) when rendering

        // cleanup
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0); // unbind vao

        delete[] positions;
        delete[] sizes;

        // create sphere model
        createSphere(1.0f, 36, 18);         // keeps sphere model in buffer for later use, removing it breaks the program
        // 36 sectors (longitude divisions)
        // 18 stacks  (latitude divisions)
    }

    // displays planets to gpu
    void renderSphere(const vec3 &position, float radius, const vec3 &color) {

        mat4 model = mat4(1.0f);            // identity matrix
        model = translate(model, position); // move to world position from origin
        model = scale(model, vec3(radius)); // scale to size

        // passes model matrix (position) and color to shaderprogram (to output to screen)
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &color[0]);

        glBindVertexArray(sphereVAO); // create vao
        glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0); // draw sphere
        glBindVertexArray(0); // unbind vao

    }

    // displays stars to gpu
    void renderStars(const mat4 &view, const mat4 &projection) {

        glUseProgram(shaderProgram); // activate shader program
        
        mat4 model = mat4(1.0f);     // identity matrix
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]); // upload to shader model

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);             // upload view matrix
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]); // upload projection matrix
        glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 1.0f, 1.0f); // star color (white - 1.0, 1.0, 1.0)

        // View Matrix: transforms world-space coordinates to camera-space coordinates, simulating a virtual camera.
        // Projection Matrix: defines how 3D coordinates are projected onto the 2D screen, including perspective distortion

        // rendering
        glBindVertexArray(starVAO);                       // binds vao
        glDrawArrays(GL_POINTS, 0, starPositions.size()); // draw stars
        glBindVertexArray(0);                             // unbind vao
    }

    // load image textures
    unsigned int loadTexture(const char *path)
    {
        // texture object
        unsigned int textureID;
        glGenTextures(1, &textureID);

        int width, height, nrChannels;
        stbi_set_flip_vertically_on_load(true); // OpenGL expects textures with (0,0) at bottom-left

        // width/height: Output variables for image dimensions (in pixels)
        // nrChannels: Output for color channels (1,3,4)
        // 0: load as is

        // load image data
        unsigned char *data = stbi_load(path, &width, &height, &nrChannels, 0); // width × height × nrChannels bytes
        if (!data) {
            std::cerr << "Failed to load texture at path: " << path << std::endl;
            return 0;
        }

        // format detection
        GLenum format = GL_RGB;
        if (nrChannels == 1)
            format = GL_RED;
        else if (nrChannels == 3)
            format = GL_RGB;
        else if (nrChannels == 4)
            format = GL_RGBA;

        // upload texture
        glBindTexture(GL_TEXTURE_2D, textureID); // active texture for GL_TEXTURE_2D
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data); // defines texture image
        
        // precomputed, downscaled versions of the texture
        glGenerateMipmap(GL_TEXTURE_2D);

        // wrapping 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // x-axis
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT); // y-axis

        // GL_TEXTURE_WRAP_S: horizontal texture wrapping
        // GL_TEXTURE_WRAP_T: vertical texture wrapping
        // GL_REPEAT: wrapping mode (tiling)

        // filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // minification (smooth downscaling)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // magnification (smooth upscaling)

        // cleanup
        stbi_image_free(data);

        return textureID;
    }

    int main()
    {

        // initialize GLFW and OpenGL version
        if (!glfwInit())
        {
            cerr << "Failed to initialize GLFW" << std::endl;
            return -1;
        }

        stbi_set_flip_vertically_on_load(true); // flip images vertically on load

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        // Create Window and rendering context using GLFW, resolution is 800x600
        GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Solar System", NULL, NULL);
        if (!window)
        {
            cerr << "Failed to create GLFW window" << endl;
            glfwTerminate();
            return -1;
        }

        glfwMakeContextCurrent(window);
        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        // initialize GLEW
        if (glewInit() != GLEW_OK)
        {
            cerr << "Failed to initialize GLEW" << endl;
            return -1;
        }

        // load planet textures
        unsigned int sunTexture = loadTexture("Textures/Sun.png");
        unsigned int mercuryTexture = loadTexture("Textures/Mercury.jpg");
        unsigned int venusTexture = loadTexture("Textures/Venus.jpg");
        unsigned int earthTexture = loadTexture("Textures/Earth.jpg");
        unsigned int marsTexture = loadTexture("Textures/Mars.jpg");
        unsigned int jupiterTexture = loadTexture("Textures/Jupiter.jpg");
        unsigned int saturnTexture = loadTexture("Textures/Saturn.jpg");
        unsigned int uranusTexture = loadTexture("Textures/Uranus.jpg");
        unsigned int neptuneTexture = loadTexture("Textures/Neptune.jpg");

        // all textures
        std::vector<GLuint> planetTextures;
        planetTextures.push_back(sunTexture);     // 0 = Sun 1
        planetTextures.push_back(sunTexture);     // 1 = Sun 2
        planetTextures.push_back(mercuryTexture); // 2 = Mercury
        planetTextures.push_back(venusTexture);   // 3 = Venus
        planetTextures.push_back(earthTexture);   // 4 = Earth
        planetTextures.push_back(marsTexture);    // 5 = Mars
        planetTextures.push_back(jupiterTexture); // 6 = Jupiter
        planetTextures.push_back(saturnTexture);  // 7 = Saturn
        planetTextures.push_back(uranusTexture);  // 8 = Uranus
        planetTextures.push_back(neptuneTexture); // 9 = Neptune

        // load moon textures
        unsigned int moonTexture = loadTexture("Textures/The moon.jpg");
        unsigned int phobosTexture = loadTexture("Textures/Phobos.jpg");
        unsigned int deimosTexture = loadTexture("Textures/Deimos.jpg");
        unsigned int ioTexture = loadTexture("Textures/Io.jpg");
        unsigned int europaTexture = loadTexture("Textures/Europa.jpg");
        unsigned int ganymedeTexture = loadTexture("Textures/Ganymede.jpg");
        unsigned int callistoTexture = loadTexture("Textures/Callisto.jpg");

        // all moon textures
        std::vector<GLuint> moonTextures;
        moonTextures.push_back(moonTexture);     // 0 = Moon
        moonTextures.push_back(phobosTexture);   // 1 = Phobos
        moonTextures.push_back(deimosTexture);   // 2 = Deimos
        moonTextures.push_back(ioTexture);       // 3 = Io
        moonTextures.push_back(europaTexture);   // 4 = Europa
        moonTextures.push_back(ganymedeTexture); // 5 = Ganymede
        moonTextures.push_back(callistoTexture); // 6 = Callisto

        // setup
        srand(static_cast<unsigned int>(time(nullptr)));    // random number generator seed
        generateStarField();                                // creates the background starfield
        initSolarSystem();                                  // initializes the planets in the solar system
        compileAndLinkShaders();                            // sets up the GLSL shaders
        GLuint satelliteShader = compileSatelliteShaders(); // satellite shader
        loadSatelliteModels();
        setupBuffers();                                     // sets up OpenGL vertex buffers

        // opengl states
        glEnable(GL_DEPTH_TEST);                           // enables depth testing so objects closer to the camera obscure farther ones
        glDepthFunc(GL_LESS);
        glEnable(GL_PROGRAM_POINT_SIZE);                   // allows the shader to control point (star) sizes
        glEnable(GL_BLEND);                                // enables alpha blending for transparency effects
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // sets the blending function for combining colors with what's already in the framebuffer

        float deltaTime = 0.0f;
        float lastFrame = 0.0f;

        while (!glfwWindowShouldClose(window))
        {
            // frame time
            float currentFrame = static_cast<float>(glfwGetTime());
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;

            // constantly update camera + planet position
            processInput(window, deltaTime);
            updatePlanets(deltaTime);

            // clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // set view/projection matrices
            mat4 view = lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
            mat4 projection = perspective(radians(45.0f),
                                        (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT,
                                        0.1f,
                                        MAX_DISTANCE * 2.0f);

            // render and update satellites
            vec3 earthPos;
            int index = 0;
            for (const auto &planet : planets) {
                if (index == 4) {
                    earthPos = planet.position;
                    break;
                }
                index++;
            }
            updateSatellites(deltaTime, earthPos);
            renderSatellites(view, projection, satelliteShader);

            // configure shader
            glUseProgram(shaderProgram);
            glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 1.0f, 1.0f); // set to white
            glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);           // ensure texture is off

            // draw stars to screen
            renderStars(view, projection);

            // shaders
            glUseProgram(shaderProgram);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);

            // pass sun positions and colors to shader
            vec3 sunPositions[2];
            vec3 sunColors[2];
            int sunIdx = 0;

            // calculates sun's position and color to pass to the shader for lighting
            for (auto it = planets.begin(); it != planets.end() && sunIdx < 2; ++it, ++sunIdx)
            {
                sunPositions[sunIdx] = it->position;
                sunColors[sunIdx] = it->color;
            }

            // pass the sun's position to the shader so it can be calculated and outputted to screen
            glUniform3fv(glGetUniformLocation(shaderProgram, "sunPositions"), 2, &sunPositions[0][0]);
            glUniform3fv(glGetUniformLocation(shaderProgram, "sunColors"), 2, &sunColors[0][0]);
            glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, &cameraPos[0]);

            // ring for saturn
            createRingVAO(0.9f, 1.3f);

            int i = 0;
            int moonTextureIndex = 0;
            for (auto it = planets.begin(); it != planets.end(); ++it, i++)
            {
                // draw planet
                Planet &planet = *it;
                mat4 model = mat4(1.0f);
                model = translate(model, planet.position);
                model = rotate(model, radians(planet.rotationAngle), vec3(0.0f, 1.0f, 0.0f));
                model = scale(model, vec3(planet.radius));

                glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);

                // set isSun uniform: first two planets are suns
                int isSun = (i < 2) ? 1 : 0;
                glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), isSun);

                // bind planet texture by index
                if (i < planetTextures.size()) {
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, planetTextures[i]);
                    glUniform1i(glGetUniformLocation(shaderProgram, "planetTexture"), 0);
                    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);
                } else {
                    glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &planet.color[0]);
                    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
                }

                // adds ring to planet
                if (planet.hasRing)
                {
                    // set ring shader parameters
                    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0); // no texture on ring
                    glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &planet.ringColor[0]);
                    glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 0);

                    // calculate ring model matrix
                    mat4 ringModel = mat4(1.0f);
                    ringModel = translate(ringModel, planet.position);
                    ringModel = rotate(ringModel, radians(planet.rotationAngle * 0.5f), vec3(0.0f, 1.0f, 0.0f)); // rotate slower than planet
                    ringModel = rotate(ringModel, radians(25.0f), vec3(1.0f, 0.0f, 0.0f));                       // ring tilt

                    // scale ring to proper size
                    float ringScale = planet.radius * 1.5f; // Adjust as needed
                    ringModel = scale(ringModel, vec3(ringScale));

                    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &ringModel[0][0]);

                    // draw the ring
                    glBindVertexArray(ringVAO);
                    glDrawElements(GL_TRIANGLES, ringIndexCount, GL_UNSIGNED_INT, 0);
                    glBindVertexArray(0);

                    // restore planet rendering state
                    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1); // re-enable textures
                }

                /*if (planet.hasRing) {
                    // Save current shader state
                    GLint prevUseTexture;
                    glGetUniformiv(shaderProgram, GL_CURRENT_PROGRAM, &prevUseTexture);

                    // Set up ring rendering
                    mat4 ringModel = mat4(1.0f);
                    ringModel = translate(ringModel, planet.position);
                    ringModel = rotate(ringModel, radians(planet.rotationAngle), vec3(0.0f, 1.0f, 0.0f)); // Rotate with planet
                    ringModel = rotate(ringModel, radians(90.0f), vec3(1.0f, 0.0f, 0.0f));                // Tilt for ring appearance

                    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &ringModel[0][0]);
                    glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &planet.ringColor[0]);
                    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0); // No texture for rings
                    glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 0);

                    // Draw the ring
                    glBindVertexArray(ringVAO);
                    glDrawElements(GL_TRIANGLES, ringIndexCount, GL_UNSIGNED_INT, 0);

                    // Restore shader state
                    glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), prevUseTexture);
                }*/

                // draw sphere
                glBindVertexArray(sphereVAO);
                glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);

                // draw moons 
                for (const auto &moon : planet.moons) {
                    mat4 moonModel = mat4(1.0f);
                    moonModel = translate(moonModel, planet.position + moon.offset);
                    moonModel = scale(moonModel, vec3(moon.radius));

                    // sends data to shader
                    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &moonModel[0][0]);
                    glUniform1i(glGetUniformLocation(shaderProgram, "isSun"), 0); // moons are never suns

                    // bind moon texture in order
                    if (moonTextureIndex < moonTextures.size()) {
                        glActiveTexture(GL_TEXTURE0);
                        glBindTexture(GL_TEXTURE_2D, moonTextures[moonTextureIndex]);
                        glUniform1i(glGetUniformLocation(shaderProgram, "planetTexture"), 0);
                        glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);
                    } else {
                        glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &moon.color[0]);
                        glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
                    }

                    // draws moon
                    glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);
                    moonTextureIndex++;
                }

                glBindVertexArray(0); // unbind vao
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
        glDeleteProgram(shaderProgram);

        glfwTerminate();
        return 0;
    }
