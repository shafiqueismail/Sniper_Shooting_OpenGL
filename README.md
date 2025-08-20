
# Team
Grayson Yen

Anis Alouache

Hassan Moharram

Ismail Shafique

# Setup:
1. Download repo (Code -> Download ZIP)
2. Launch Ubuntu
3. Install Dependencies 
4. Run "code ." in Ubuntu terminal to launch VSCode in WSL.
5. Unzip repo folder to WSL folder.
6. Open and launch stars.cpp using g++ or use g++ compiler in VSCode.

# Movement
W - forward

A - left

S - back

D - right

Left CTRL - downwards

Space - upwards

ESC - exit

C - Sniper Scope

Z - Double Barrel Gun

X - Grenade

L - Lazer bullet

# Mouse
Camera follows mouse movement. Direction is relative to the camera, therefore movement follows where the mouse moves.

# How it works
## Planets
We define a function (`initSolarSystem()`) that generates the planets of our solar system. There's two suns that orbit in a circular pattern. We also define a struct called `Planet` that allows to model the planets in our solar system. 

For orbiting, we calculate a planet's current position as follows: 
`x = orbitRadius * cos(angle), z = orbitRadius * sin(angle)` 
We only use x and z since the planets/suns do not move upwards. The moons are calculated in the same way, but using an offset as well to ensure there is a gap between the moon and the planet, but it still follows its trajectory. 

Each planet has a unique orbit speed and starting angle, since each planet has different times to orbit around the sun. Additionally, each planet that has a moon rotates around its planet.

We render a planet using `createSphere()`, and we update its position/angle as well as its moon(s)' position at every instance using `updatePlanets()`. Additionally, we also load in textures (taken from the NASA website) and update shaders (the lighting from the suns). 

For the lighting, we pass data from the suns' positions to the shader:

`glUniform3fv(glGetUniformLocation(shaderProgram, "sunPositions"), 2, &sunPositions[0][0]);`

`glUniform3fv(glGetUniformLocation(shaderProgram, "sunColors"), 2, &sunColors[0][0]);`

`glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, &cameraPos[0]);`

where the lighting is calculated and outputted to the screen.


## Stars
We define a function (`generateStarField()`) that generates 2000 points across the space centered around the origin. The points are unordered and varying in sizes (with them skewed towards smaller sizes). This gives the impression of natural stars in space. 

`NUM_STARS = 2000`: The total number of stars to generate

`MAX_DISTANCE = 200.0f`: The maximum distance from origin where stars can appear

`MIN_SIZE = 0.02f` and `MAX_SIZE = 0.15f`: The range of star sizes

**Position Generation**:

Each star is placed at a random distance from the origin (0,0,0). The distance distribution is non-linear (using `pow(rand(), 0.25f)`), this allows us to create more stars closer to the origin.

Spherical coordinates (theta and phi) are used to position stars in 3D space:

`float theta = 2.0f * pi<float>() * static_cast<float>(rand()) / RAND_MAX;  
float phi = acos(2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);`

Then, we convert from spherical coordinates to Cartesian coordinates to match our space:

`
float x = distance * sin(phi) * cos(theta);
float y = distance * sin(phi) * sin(theta);
float z = distance * cos(phi);
`

**We render the stars as follows:**

The positions of each star is obtained from the `starVBO` buffer.

The size of each star is obtained from the `starSizeVBO` buffer.

Using `GL_POINTS` primitive type, we can draw the stars: `glDrawArrays(GL_POINTS, 0, starPositions.size());`


## Rotation
As mentioned in Planets, we calculate a planet's current position as follows: 
`x = orbitRadius * cos(angle), z = orbitRadius * sin(angle)` 

Since the planets do not travel upwards, we do not need to calculate the y-axis. 
We take the radius of the planet, calculate its direction using its current angle (meaning, which direction the planet is moving), which then allows us to calculate where the planet will be in the next instance. We do this for every planet, moon, and sun at each moment the program is running.

### Bibiliography
In terms of sources when building the code, the following were used: Google, tutorial notes and chatgpt.
