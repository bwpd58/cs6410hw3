// CS6410 HW3 Bryce Paubel 3/30/22
// This program renders an object through multiple framebuffers to 
// draw a rainbow map of the light intensity on an object, and allows
// the user to select multiple lighting shaders to see how the light intensity changes

// HEAVILY BASED ON TOON SHADER EXAMPLE
// ANY CHANGES ARE MARKED /***** MAJOR EDIT *****/

/* NOTE - I resized this project to fit the window */

// .obj viewer
// draws on custom framebuffer that is then used as texture

/***** MAJOR EDIT *****/
// Creating a Gouraud shader with point lighting and no specular highlights
const vertex_gouraud_no_specular = `#version 300 es
    in vec4 a_position; // in instead of attribute
    in vec4 a_color; // in instead of attribute
    in vec4 a_normal; // in instead of attribute
    in vec2 a_texcoord; // in instead of attribute
    in vec3 a_barycoord; // barycentric coordinates

    uniform mat4 u_mvp_mat;
    uniform mat4 u_normal_mat;

    out vec4 v_color; // out instead of varying
    out vec2 v_texcoord; // out instead of varying
    out vec3 v_barycoord; // out instead of varying

    void main() {
        vec3 normal = normalize(vec3(u_normal_mat * a_normal));
        v_texcoord = a_texcoord;
        v_barycoord = a_barycoord;

        vec3 eye_position = vec3(0.0, 20.0, 40.0); 
        vec3 light_position = vec3(0.0, 20.0, 40.0); // same as camera

        float shininess = 50.0;

        // Calculate the light direction and make it 1.0 in length
        vec3 light_dir = normalize(light_position - a_position.xyz);

        // The dot product of the light direction and the normal
        vec3 diffuse = max(dot(light_dir, normal), 0.0) * a_color.rgb;

        // Calculate the eye direction and make it 1.0 in length
        vec3 eye_dir = normalize(eye_position - a_position.xyz);

        // The halfway vector between light_dir and eye_dir
        vec3 halfway = (light_dir + eye_dir) / length(light_dir + eye_dir);

        //vec3 specular = pow(max(dot(normal, halfway), 0.0), shininess) * a_color.rgb;

        v_color = vec4(diffuse, 1.0);

        gl_Position = u_mvp_mat * a_position;
    }
`;

/***** MAJOR EDIT *****/
// Creating a Gouraud shader with point lighting and specular highlights
const vertex_gouraud_specular = `#version 300 es
    in vec4 a_position; // in instead of attribute
    in vec4 a_color; // in instead of attribute
    in vec4 a_normal; // in instead of attribute
    in vec2 a_texcoord; // in instead of attribute
    in vec3 a_barycoord; // barycentric coordinates

    uniform mat4 u_mvp_mat;
    uniform mat4 u_normal_mat;

    out vec4 v_color; // out instead of varying
    out vec2 v_texcoord; // out instead of varying
    out vec3 v_barycoord; // out instead of varying

    void main() {
        vec3 normal = normalize(vec3(u_normal_mat * a_normal));
        v_texcoord = a_texcoord;
        v_barycoord = a_barycoord;

        vec3 eye_position = vec3(0.0, 20.0, 40.0); 
        vec3 light_position = vec3(0.0, 20.0, 40.0); // same as camera

        float shininess = 50.0;

        // Calculate the light direction and make it 1.0 in length
        vec3 light_dir = normalize(light_position - a_position.xyz);

        // The dot product of the light direction and the normal
        vec3 diffuse = max(dot(light_dir, normal), 0.0) * a_color.rgb;

        // Calculate the eye direction and make it 1.0 in length
        vec3 eye_dir = normalize(eye_position - a_position.xyz);

        // The halfway vector between light_dir and eye_dir
        vec3 halfway = (light_dir + eye_dir) / length(light_dir + eye_dir);

        vec3 specular = pow(max(dot(normal, halfway), 0.0), shininess) * a_color.rgb;

        v_color = vec4(diffuse + specular, 1.0);

        gl_Position = u_mvp_mat * a_position;
    }
`;


const frag_gouraud = `#version 300 es
    precision mediump float;

    uniform sampler2D u_image;
    uniform bool u_is_texture;
    in vec2 v_texcoord;
    in vec3 v_barycoord;

    in vec4 v_color; // in instead of varying
    out vec4 cs_FragColor; // user-defined instead of gl_FragColor
    
    void main() {
    	vec4 cout;
        if (u_is_texture) {
            vec3 c = texture(u_image, v_texcoord).rgb; // texel color
            c = c * v_color.rgb; // obj shaded and textured
            cout = vec4(c, 1.0);
        }
        else cout = v_color;
        float g = (cout.r + cout.g + cout.b) / 3.0;
        cout = vec4(g, g, g, 1.0);
        cs_FragColor = cout;
    }
`;

/***** MAJOR EDIT *****/
// Add a Phong shader which uses 1 - average of rgb to determine rainbow color
const frag_phong_rainbow_specular = `#version 300 es
    precision mediump float;

    uniform sampler2D u_image;
    uniform bool u_is_texture;
    in vec2 v_texcoord;
    in vec3 v_barycoord;

    in vec4 v_color; 
    in vec3 v_normal; 
    in vec3 v_position;

    out vec4 cg_FragColor; // user-defined instead of gl_FragColor

	// CODE FROM saturate.js
	vec3 rgb2hsv(vec3 c) {
	    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
	    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
	
	    float d = q.x - min(q.w, q.y);
	    float e = 1.0e-10;
	    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
	}

	// CODE FROM saturate.js
	vec3 hsv2rgb(vec3 c) {
	    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
	}

    void main() {
        vec4 cout;
        if (u_is_texture) {
            vec3 c = texture(u_image, v_texcoord).rgb; // texel color
            c = c * v_color.rgb; // obj shaded and textured
            cout = vec4(c, 1.0);
        }
        else cout = v_color;  
        
        // Normalize the normal because it is interpolated and not 1.0 in length any more
        vec3 normal = normalize(v_normal);

        vec3 eye_position = vec3(0.0, 20.0, 40.0); 
        vec3 light_position = vec3(0.0, 20.0, 40.0); // same as camera

        //float shininess = 50.0;
        float shininess = 10.0;

        // Calculate the light direction and make it 1.0 in length
        vec3 light_dir = normalize(light_position - v_position);

        // The dot product of the light direction and the normal
        vec3 diffuse = max(dot(light_dir, normal), 0.0) * cout.rgb;

        // Calculate the eye direction and make it 1.0 in length
        vec3 eye_dir = normalize(eye_position - v_position);

        // The halfway vector between light_dir and eye_dir
        vec3 halfway = (light_dir + eye_dir) / length(light_dir + eye_dir);

        vec3 specular = pow(max(dot(normal, halfway), 0.0), shininess) * cout.rgb;

        cout = vec4(diffuse + specular, 1.0);

		/***** MAJOR EDIT *****/
		// Determine grayscale and find the hue of that particular grayscale
		float average = (cout.r + cout.g + cout.b) / 3.0;

        cg_FragColor = vec4(hsv2rgb(vec3(1.0 - average, 1.0, 1.0)), 1.0);
    }
`;

/***** MAJOR EDIT *****/
// Add a Phong shader (no specular) which uses 1 - average of rgb to determine rainbow color
const frag_phong_rainbow_no_specular = `#version 300 es
    precision mediump float;

    uniform sampler2D u_image;
    uniform bool u_is_texture;
    in vec2 v_texcoord;
    in vec3 v_barycoord;

    in vec4 v_color; 
    in vec3 v_normal; 
    in vec3 v_position;

    out vec4 cg_FragColor; // user-defined instead of gl_FragColor

	// CODE FROM saturate.js
	vec3 rgb2hsv(vec3 c) {
	    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
	    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
	
	    float d = q.x - min(q.w, q.y);
	    float e = 1.0e-10;
	    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
	}

	// CODE FROM saturate.js
	vec3 hsv2rgb(vec3 c) {
	    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
	}

    void main() {
        vec4 cout;
        if (u_is_texture) {
            vec3 c = texture(u_image, v_texcoord).rgb; // texel color
            c = c * v_color.rgb; // obj shaded and textured
            cout = vec4(c, 1.0);
        }
        else cout = v_color;  
        
        // Normalize the normal because it is interpolated and not 1.0 in length any more
        vec3 normal = normalize(v_normal);

        vec3 eye_position = vec3(0.0, 20.0, 40.0); 
        vec3 light_position = vec3(0.0, 20.0, 40.0); // same as camera

        //float shininess = 50.0;
        float shininess = 10.0;

        // Calculate the light direction and make it 1.0 in length
        vec3 light_dir = normalize(light_position - v_position);

        // The dot product of the light direction and the normal
        vec3 diffuse = max(dot(light_dir, normal), 0.0) * cout.rgb;

        // Calculate the eye direction and make it 1.0 in length
        vec3 eye_dir = normalize(eye_position - v_position);

        // The halfway vector between light_dir and eye_dir
        vec3 halfway = (light_dir + eye_dir) / length(light_dir + eye_dir);

        // vec3 specular = pow(max(dot(normal, halfway), 0.0), shininess) * cout.rgb;

        cout = vec4(diffuse, 1.0);

		/***** MAJOR EDIT *****/
		// Determine grayscale and find the hue of that particular grayscale
		float average = (cout.r + cout.g + cout.b) / 3.0;

        cg_FragColor = vec4(hsv2rgb(vec3(1.0 - average, 1.0, 1.0)), 1.0);
    }
`;

/***** MAJOR EDIT *****/
// Create a rainbow Gouraud shader
// This fragment shader uses 1 - rbg average to determine light intensity based on the color passed to it
const frag_gouraud_rainbow = `#version 300 es
    precision mediump float;

    uniform sampler2D u_image;
    uniform bool u_is_texture;
    in vec2 v_texcoord;
    in vec3 v_barycoord;

    in vec4 v_color; // in instead of varying
    out vec4 cs_FragColor; // user-defined instead of gl_FragColor

	// CODE FROM saturate.js
	vec3 rgb2hsv(vec3 c) {
	    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
	    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
	
	    float d = q.x - min(q.w, q.y);
	    float e = 1.0e-10;
	    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
	}

	// CODE FROM saturate.js
	vec3 hsv2rgb(vec3 c) {
	    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
	}
    
    void main() {
    	vec4 cout;
        if (u_is_texture) {
            vec3 c = texture(u_image, v_texcoord).rgb; // texel color
            c = c * v_color.rgb; // obj shaded and textured
            cout = vec4(c, 1.0);
        }
        else cout = v_color;

		/***** MAJOR EDIT *****/
		// Determine grayscale and find the hue of that particular grayscale
		float average = (v_color.r + v_color.g + v_color.b) / 3.0;

        cs_FragColor = vec4(hsv2rgb(vec3(1.0 - average, 1.0, 1.0)), 1.0);
    }
`;

const vertex_toon = `#version 300 es
    in vec4 a_position; // in instead of attribute
    in vec4 a_color; // in instead of attribute
    in vec4 a_normal; // in instead of attribute
    in vec2 a_texcoord; // in instead of attribute
    in vec3 a_barycoord; // barycentric coordinates

    uniform mat4 u_mvp_mat;
    uniform mat4 u_normal_mat;
    uniform mat4 u_model_mat;
    
    out vec2 v_texcoord; // out instead of varying
    out vec3 v_barycoord; // out instead of varying

    out vec4 v_color; // out instead of varying
    out vec3 v_normal; 
    out vec3 v_position;

    void main() {
        v_position = vec3(u_model_mat * a_position);
        v_normal = normalize(vec3(u_normal_mat * a_normal));

        v_color = a_color;

        v_texcoord = a_texcoord;
        v_barycoord = a_barycoord;

        gl_Position = u_mvp_mat * a_position;
    }
`;

const frag_toon = `#version 300 es
    precision mediump float;

    uniform sampler2D u_image;
    uniform bool u_is_texture;
    in vec2 v_texcoord;
    in vec3 v_barycoord;

    in vec4 v_color; 
    in vec3 v_normal; 
    in vec3 v_position;

    out vec4 cg_FragColor; // user-defined instead of gl_FragColor

    void main() {
        vec4 cout;
        if (u_is_texture) {
            vec3 c = texture(u_image, v_texcoord).rgb; // texel color
            c = c * v_color.rgb; // obj shaded and textured
            cout = vec4(c, 1.0);
        }
        else cout = v_color;  
        
        // Normalize the normal because it is interpolated and not 1.0 in length any more
        vec3 normal = normalize(v_normal);

        vec3 eye_position = vec3(0.0, 20.0, 40.0); 
        vec3 light_position = vec3(0.0, 20.0, 40.0); // same as camera

        //float shininess = 50.0;
        float shininess = 10.0;

        // Calculate the light direction and make it 1.0 in length
        vec3 light_dir = normalize(light_position - v_position);

        // The dot product of the light direction and the normal
        vec3 diffuse = max(dot(light_dir, normal), 0.0) * cout.rgb;

        // Calculate the eye direction and make it 1.0 in length
        vec3 eye_dir = normalize(eye_position - v_position);

        // The halfway vector between light_dir and eye_dir
        vec3 halfway = (light_dir + eye_dir) / length(light_dir + eye_dir);

        vec3 specular = pow(max(dot(normal, halfway), 0.0), shininess) * cout.rgb;

        cout = vec4(diffuse + specular, 1.0);

        float g = (cout.r + cout.g + cout.b) / 3.0;
        
        if (g > 0.66) g = 1.0;
        else if (g > 0.33) g = 0.66;
        else g = 0.33;
        cout = vec4(0.0, g, g, 1.0); // cyan 

        cout = clamp(cout, 0.0, 1.0);
        
        //cg_FragColor = vec4(diffuse + specular, cout.a);
        cg_FragColor = cout;
    }
`;

// computes v_coord here; no need to receive texcoord array as attribute variable
const vertex_display = `#version 300 es
	in vec2 a_position;	
	out vec2 v_coord;

	void main() {	   
	   gl_PointSize = 1.0;
	   gl_Position = vec4(a_position, 0.0, 1.0); // 4 corner vertices of quad

	   v_coord = a_position * 0.5 + 0.5; // UV coords: (0, 0), (0, 1), (1, 1), (1, 0)
	}
`;

const frag_display = `#version 300 es
	precision mediump float;
	precision highp sampler2D;

	uniform sampler2D u_image;
	in vec2 v_coord;

	out vec4 cg_FragColor; 

	void main() {
	   cg_FragColor = texture(u_image, v_coord);
	}
`;

const frag_gauss = `#version 300 es
    precision highp float; 
    precision highp sampler2D;
    
    uniform vec2 u_texel; // added by hk
    uniform sampler2D u_image;
    uniform float u_half; // kernel half width
    in vec2 v_coord;
    out vec4 cg_FragColor;
        
    void main () {
        float x = v_coord.x;
        float y = v_coord.y;
        float dx = u_texel.x;
        float dy = u_texel.y;        

		float sigma = u_half; 
        float twoSigma2 = 2.0 * sigma * sigma;
        vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
        float w_sum = 0.0;
        	
        for (float j = -u_half; j <= u_half; j+=1.0) {	
			for (float i = -u_half; i <= u_half; i+=1.0) {	
			    float d = distance(vec2(0.0), vec2(i, j));
			    if (d > u_half) continue;		
				float weight = exp(-d * d / twoSigma2);
				vec4 st = texture(u_image, vec2(x+dx*i, y+dy*j));
				sum += weight * st; // sum is float4
				w_sum += weight;
			}
        }		
		
		sum /= w_sum; // normalize weight
		                
	    cg_FragColor = sum; 
    }
`;

// assuming grayscale input image
const frag_sobel = `#version 300 es
	precision highp float; 
    precision highp sampler2D;
    
    in vec2 v_coord;
    uniform sampler2D u_image;
    uniform vec2 u_texel; // added by hk
    uniform float u_scale;

    out vec4 cg_FragColor;
        
    void main () {
        float x = v_coord.x;
        float y = v_coord.y;
        float dx = u_texel.x;
        float dy = u_texel.y;    
  
        vec2 g = vec2(0.0);

        g.x = (          
			-1.0 * texture(u_image, vec2(x-dx, y-dy)).r +
			-2.0 * texture(u_image, vec2(x-dx, y)).r +
			-1.0 * texture(u_image, vec2(x-dx, y+dy)).r +
			+1.0 * texture(u_image, vec2(x+dx, y-dy)).r +
			+2.0 * texture(u_image, vec2(x+dx, y)).r +
			+1.0 * texture(u_image, vec2(x+dx, y+dy)).r		
		); // [-4, 4] because texture returns [0, 1]		   

		g.y = (		
			-1.0 * texture(u_image, vec2(x-dx, y-dy)).r +
			-2.0 * texture(u_image, vec2(x,    y-dy)).r +
			-1.0 * texture(u_image, vec2(x+dx, y-dy)).r +
			+1.0 * texture(u_image, vec2(x-dx, y+dy)).r +
			+2.0 * texture(u_image, vec2(x,    y+dy)).r +
			+1.0 * texture(u_image, vec2(x+dx, y+dy)).r		
		); // [-4, 4] because texture returns [0, 1]		   
		
		g.x /= 4.0; // [-1, 1]
	    g.y /= 4.0; // [-1, 1]	
	    
		float mag = g.x * g.x + g.y * g.y; // [0, 2]
	    mag /= 2.0; // [0, 1]	   	   

        // if zero gradient, make it a vertical tangent vector
        if (g.x == 0.0 && g.y == 0.0) g = vec2(1.0, 0.0); 
	    
	    g = normalize(g); // [-1, 1]

	    g = (g + 1.0) / 2.0; // [0, 1]

        /////////////////////////////////////////
        // enhance gradient magnitude
	    mag = tanh(u_scale * mag);
	    /////////////////////////////////////////

	    //cg_FragColor = vec4(mag, mag, mag, 1.0);
	    cg_FragColor = vec4(g, 0.0, mag);	   
    }       
`;

const frag_gray = `#version 300 es
	precision highp float;
	precision highp sampler2D;

	uniform sampler2D u_image;
	in vec2 v_coord;
	out vec4 cg_FragColor;

	void main() {
	   vec4 c = texture(u_image, v_coord); // [0, 1]
	   float g = (c.r + c.g + c.g) / 3.0; // [0, 1]
	   cg_FragColor = vec4(g, g, g, 1.0);
	}
`;

const frag_gradient_2_mag = `#version 300 es	
    precision highp float; 
    precision highp sampler2D;

    in vec2 v_coord;
    uniform sampler2D u_gradient; // gradient 

    out vec4 cg_FragColor;
	 
    void main() {     	        

        vec4 g = texture(u_gradient, v_coord);
		                
	    cg_FragColor = vec4(g.a, g.a, g.a, 1.0);	    	    
    } 
`;

const frag_nonmaxima_suppression = `#version 300 es
    precision highp float; 
    precision highp sampler2D;
    
    uniform vec2 u_texel; 
    uniform sampler2D u_gradient;
    uniform sampler2D u_mag;
    uniform float u_thres;

    in vec2 v_coord;
    out vec4 cg_FragColor;
        
    void main () {
        float m = texture(u_mag, v_coord).r;
        vec2 g = texture(u_gradient, v_coord).xy;
        g = (g * 2.0) - 1.0; // [-1, 1]
        vec2 coord_1 = v_coord + g * u_texel;
        vec2 coord_2 = v_coord - g * u_texel;
        float m1 = texture(u_mag, coord_1).r;
        float m2 = texture(u_mag, coord_2).r;

        vec4 cout = vec4(1.0);

        if (m > m1 && m > m2 && m > u_thres)
            cout = vec4(0.0, 0.0, 0.0, 1.0); // maximum - black
        
        cg_FragColor = cout;
        //cg_FragColor = vec4(1.0, 1.0, 0.0, 1.0);
    }
`;

const frag_dilation = `#version 300 es
    precision highp float; 
    precision highp sampler2D;

    in highp vec2 v_coord;
    uniform vec2 u_texel; 
    uniform sampler2D u_image;
    uniform float u_radius;
    
    out vec4 cg_FragColor;
        
    void main () {
        float x = v_coord.x;
        float y = v_coord.y;
        float dx = u_texel.x;
        float dy = u_texel.y;        

		//vec4 cout = vec4(0.0);
		//vec4 cout = vec4(1.0);
		vec4 cout = vec4(1.0, 1.0, 1.0, 0.0); // alpha = 0.0
        		
        for (float s = -u_radius; s <= u_radius; s+=1.0) {
        	for (float t = -u_radius; t <= u_radius; t+=1.0) {
				vec4 c = texture(u_image, vec2(x+dx*s, y+dy*t));
				if (c.r < 0.5) { // black pixel found
					float d = sqrt(s*s + t*t);
					if (d <= u_radius) cout = c;
				}
        	}
		}			
               
	    cg_FragColor = cout;
    }
`;

const frag_blend_src_alpha = `#version 300 es
	precision mediump float;
	precision highp sampler2D;

	in vec2 v_coord;

	uniform sampler2D u_src;
	uniform sampler2D u_dst;

	out vec4 cg_FragColor;
	
	void main() {
		vec4 s = texture(u_src, v_coord);
		vec4 d = texture(u_dst, v_coord);
		float alpha = s.a;
		vec3 c = alpha * s.rgb + (1.0 - alpha) * d.rgb;

		cg_FragColor = vec4(c, 1.0);
	}
`;

	
let config = {
    SPEED_X: 0.0,
    SPEED_Y: 0.0,
    CAMERA_DIST: 30,
    SHADER: 0,
}

let url_prefix = "http://www.cs.umsl.edu/~kang/htdocs/models/";

//let url = url_prefix + "cube.obj";
//let url = url_prefix + "cube2.obj";
//let url = url_prefix + "snowman.obj";
//let url = url_prefix + "f-16.obj";
//let url = url_prefix + "f16.obj";
//let url = url_prefix + "cruiser.obj";
let url = url_prefix + "suzanne.obj";
//let url = url_prefix + "utah-teapot.obj";
//let url = url_prefix + "Knot.obj";
//let url = url_prefix + "crate.obj";
//let url = url_prefix + "bunny.obj";
//let url = url_prefix + "bunny2.obj";
//let url = url_prefix + "bunny3.obj";
//let url = url_prefix + "dragon.obj";
//let url = url_prefix + "buddha.obj";
//let url = url_prefix + "armadillo.obj";
//let url = url_prefix + "tyra.obj";
//let url = url_prefix + "brain.obj";
//let url = url_prefix + "heart.obj";

let gl, canvas;
let g_obj = null;
// The information of OBJ file
let g_data = null;
// The data needed for drawing 3D model
let vao_obj = null; // vertex array object for obj
let g_texture = [];
let g_image = [];
//let g_cur_angle = 0.0;
let g_vp_mat;
let g_anim_id;
let g_prog = []; // shader programs
let out_depth;
let out;
let toon;
let gradient;
let mag;
let canny;
let vao_image; // vao for drawing image (using 2 triangles)
let prog_display;
let prog_gauss;
let prog_gray;
let prog_sobel;
let prog_gradient_2_mag;
let prog_dilation;
let prog_toon;
let prog_gouraud;
let prog_blend_src_alpha;

/***** MAJOR EDIT *****/
// Insert canvas scaling elements
let canvas_scale_x = 0.99;
let canvas_scale_y = 0.92;

function render() {

    cancelAnimationFrame(g_anim_id); // to avoid duplicate requests
    
    g_vp_mat = new Matrix4();
    //vp_mat.setPerspective(30.0, canvas.width / canvas.height, 0.1, 500.0);
    g_vp_mat.setPerspective(30.0, canvas.width / canvas.height, 0.1, 500.0);        

    let cam_pos = calc_camera_pos();
    //g_vp_mat.lookAt(cam_pos.x, cam_pos.y, cam_pos.z, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0);
    g_vp_mat.lookAt(cam_pos.x, cam_pos.y, cam_pos.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // the first few iterations of update() may display nothing because
    // Ajax requests for .obj and .mtl may not have been returned yet! 
    var update = function() {
		
        if (g_obj != null && g_obj.mtl_ready() && g_obj.tex_ready()) {
        //if (g_obj != null && g_obj.mtl_ready()) {
            // .obj file is parsed and not null
            // all .mtl files are parsed and ready
            // this check is needed because .obj and .mtl are Ajax requests

            vao_obj_create();
            
            g_obj = null; // data already sent to GPU. Don't do it again.
        }

		let prev = out;

        if (vao_obj) { // vao defined
			/***** MAJOR EDIT *****/
			// Render depending on the shader
			if (config.SHADER == 0) {
				render_obj(prog_rainbow_phong_specular, out_depth_rainbow);
			} else if (config.SHADER == 1) {
				render_obj(prog_rainbow_phong_no_specular, out_depth_rainbow);
			} else if (config.SHADER == 2) {
				render_obj(prog_rainbow_gouraud_specular, out_depth_rainbow);
			} else {
				render_obj(prog_rainbow_gouraud_no_specular, out_depth_rainbow);
			}
			
            render_img(out_depth_rainbow, out_rainbow);
			
			let prev = out;

            render_obj(prog_gouraud, out_depth);
            render_img(out_depth, out);
            gauss(out, 2.0); // default: 1.5
            gray(out);
            sobel(out, gradient, 100); // normalized gradient
            gradient_2_mag(gradient, mag); 
            gauss(mag, 1.0); 
            nonmaxima_suppression(gradient, mag, canny, 0.05);
            dilation(canny, canny, 1);
			blend_src_alpha(canny, out_rainbow, out);
			if (config.SPEED_X > 0.001) {
				blend_src_alpha(prev, out, out);
			}	        
            render_null(out);
        }
        
        g_anim_id = requestAnimationFrame(update);
    };
    update();
}

function render_img (src, dst) {
    let program = prog_display;
    program.bind();

    if (src.single) gl.uniform1i(program.uniforms.u_image, src.attach(8));
    else gl.uniform1i(program.uniforms.u_image, src.read.attach(8));
    
    //gl.viewport(0, 0, src.width, src.height);
    gl.viewport(0, 0, dst.width, dst.height);
 
    if (dst.single) draw_vao_image(dst.fbo);
    else {
        draw_vao_image(dst.write.fbo);
        dst.swap();
    }  
}

function render_obj (prog, dst) {
    prog.bind(); // switch to different shader program
    
    gl.viewport(0, 0, canvas.width, canvas.height);

    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.write.fbo);
    
    gl.bindVertexArray(vao_obj); 
    draw_obj(prog, g_vp_mat);
    gl.bindVertexArray(null);

    dst.swap();
}

function vao_obj_create () {
    vao_obj = gl.createVertexArray();
    gl.bindVertexArray(vao_obj); 
    // start recording buffer object data

    // Prepare empty buffer objects for vertex coordinates, colors, and normals            
    let buffer_objects = init_buffer_objects(prog_toon);
    // buffer_objects is JavaScript Object containing multiple buffer objects

    //g_data = send_buffer_data(buffer_objects, g_obj);
    g_data = g_obj.get_data();
    send_buffer_data(buffer_objects);
    // call bufferData(...) to send vertices, normals, colors, indices to GPU 

    gl.bindVertexArray(null);
    // stop recording buffer object data
}

function cg_init_shaders(gl, vshader, fshader) {
  var program = createProgram(gl, vshader, fshader); // defined in cuon-utils.js

  return program;
}

class GLProgram {
    constructor (vertex_shader, frag_shader) {
        this.attributes = {};
        this.uniforms = {};
        this.program = gl.createProgram();

        this.program = cg_init_shaders(gl, vertex_shader, frag_shader);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
            throw gl.getProgramInfoLog(this.program);
        
        // register attribute variables
        const attribute_count = gl.getProgramParameter(this.program, gl.ACTIVE_ATTRIBUTES);
        for (let i = 0; i < attribute_count; i++) {
            const attribute_name = gl.getActiveAttrib(this.program, i).name;
            this.attributes[attribute_name] = gl.getAttribLocation(this.program, attribute_name);
        }

        // register uniform variables
        const uniform_count = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < uniform_count; i++) {
            const uniform_name = gl.getActiveUniform(this.program, i).name;
            this.uniforms[uniform_name] = gl.getUniformLocation(this.program, uniform_name);
        }
    }

    bind () {
        gl.useProgram(this.program);
    }
}

function main () {
    // Retrieve <canvas> element
    canvas = document.getElementById('canvas');

	/***** MAJOR EDITS ******/
	// Resizing the canvas
	canvas.width = window.innerWidth * canvas_scale_x;
	canvas.height = window.innerHeight * canvas_scale_y;
	
    // Get the rendering context for WebGL
    gl = canvas.getContext('webgl2');

    config.IMAGE_X = canvas.width;
    config.IMAGE_Y = canvas.height;

    // Initialize shaders    
    prog_toon = new GLProgram(vertex_toon, frag_toon);
    prog_toon.bind();

    prog_gouraud = new GLProgram(vertex_gouraud_no_specular, frag_gouraud);

    prog_display = new GLProgram(vertex_display, frag_display);
    // shader to draw on custom framebuffer
    prog_gauss = new GLProgram(vertex_display, frag_gauss);
    prog_sobel = new GLProgram(vertex_display, frag_sobel);
    prog_gradient_2_mag = new GLProgram(vertex_display, frag_gradient_2_mag);
    prog_nonmaxima_suppression = new GLProgram(vertex_display, frag_nonmaxima_suppression);
    prog_dilation = new GLProgram(vertex_display, frag_dilation);
    prog_blend_src_alpha = new GLProgram(vertex_display, frag_blend_src_alpha);
    prog_gray = new GLProgram(vertex_display, frag_gray);

	/***** MAJOR EDIT *****/
	// Creating the rainbow program and its variations
	// Note that all these shaders use point lighting, not directional lighting
	prog_rainbow_phong_specular = new GLProgram(vertex_toon, frag_phong_rainbow_specular);
	prog_rainbow_phong_no_specular = new GLProgram(vertex_toon, frag_phong_rainbow_no_specular)
	prog_rainbow_gouraud_specular = new GLProgram(vertex_gouraud_specular, frag_gouraud_rainbow);
	prog_rainbow_gouraud_no_specular = new GLProgram(vertex_gouraud_no_specular, frag_gouraud_rainbow)
	
    
    vao_image_create();     
    cg_init_framebuffers();

    cg_register_event_handlers();
    // Set the clear color and enable the depth test
    //gl.clearColor(0.2, 0.2, 0.2, 1.0);
    gl.clearColor(0.8, 0.8, 0.8, 1.0);
    //gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.enable(gl.DEPTH_TEST);

    // Start reading the OBJ file
    //get_obj_file(url, 0.27, true); // utah-teapot.obj
    //get_obj_file(url, 0.3, true); // utah-teapot.obj
    //get_obj_file(url, 1, true);
    //get_obj_file(url, 2, true);
    //get_obj_file(url, 3, true);
    get_obj_file(url, 5, true);
    //get_obj_file(url, 10, true);

	/***** MAJOR EDITS ******/
	// Inserting dat.GUI elements
	let gui = new dat.GUI( { width: 400 } );
    gui.add(config, "SHADER", { "Phong Rainbow (specular)": 0, "Phong Rainbow (no specular)": 1, "Gouraud Rainbow (specular)": 2, "Gouraud Rainbow (no specular)": 3 }).name("Shader").onFinishChange(render);

    render();
}

// Create buffer objects and store them in Object
function init_buffer_objects (program) {

    var o = new Object();
    // Utilize JavaScript Object object to return multiple buffer objects
    console.log(program);

    o.vertex_buffer = create_empty_buffer_object(program.attributes.a_position, 3, gl.FLOAT);
    o.normal_buffer = create_empty_buffer_object(program.attributes.a_normal, 3, gl.FLOAT);
    o.texcoord_buffer = create_empty_buffer_object(program.attributes.a_texcoord, 2, gl.FLOAT);
    o.barycoord_buffer = create_empty_buffer_object(program.attributes.a_barycoord, 3, gl.FLOAT);
    o.color_buffer = create_empty_buffer_object(program.attributes.a_color, 4, gl.FLOAT);
    o.index_buffer = gl.createBuffer();

    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    return o;
}

// Create a buffer object, assign it to attribute variables, and enable the assignment
function create_empty_buffer_object (a_attribute, num, type) {
    
    var buffer = gl.createBuffer();
    // Create a buffer object

    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.vertexAttribPointer(a_attribute, num, type, false, 0, 0);
    // Assign the buffer object to the attribute variable
    gl.enableVertexAttribArray(a_attribute);
    // Enable the assignment

    return buffer;
}

// get a file from server
//function get_obj_file (obj_filename, gl, buffer_objects, scale, reverse) {
function get_obj_file (obj_filename, scale, reverse) {

    var request = new XMLHttpRequest(); // create Ajax request 

    // even handler function to handle server's response 
    request.onreadystatechange = function() {
        if (request.readyState === 4 && request.status !== 404) {
            // readySate = 4 means process comlete
            // file access successful and ready 
            read_obj_file(request.responseText, obj_filename, scale, reverse);
            // request.responseText contains file content (long string)
        }
    }

    request.open('GET', obj_filename, true);
    // Create a request to acquire .obj file
    request.send();
    // Send the request
}

// Ajax for .obj file returned. Now let's parse .obj file
function read_obj_file (file_string, obj_filename, scale, reverse) {

    g_obj = new OBJ(obj_filename);
    // Create an OBJ object
    var result = g_obj.parse(file_string, scale, reverse);
    // Parse the .obj file
    
    if (!result) { // parse error
        g_obj = null;
        g_data = null;
        console.log("OBJ file parsing error.");
        return;
    }
}

// matrices
var g_model_mat = new Matrix4();
var g_mvp_mat = new Matrix4();
var g_normal_mat = new Matrix4();
let right_pos = new Vector4([1, 0, 0, 1]); // right pos (rotated by y-roll)
let cur_right_pos = new Vector4([1, 0, 0, 1]); // right pos (rotated by y-roll)
let y_roll_mat = new Matrix4();
let inv_y_roll_mat = new Matrix4();

function draw_obj (program, vp_mat) {

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    // Clear color and depth buffers
    
    config.SPEED_X *= 0.9;
    config.SPEED_Y *= 0.9;
    let p = cur_right_pos.elements;
    g_model_mat.rotate(config.SPEED_X, p[0], p[1], p[2]);
    g_model_mat.rotate(config.SPEED_Y, 0, 1, 0);
    y_roll_mat.rotate(config.SPEED_Y, 0, 1, 0);
    inv_y_roll_mat.setInverseOf(y_roll_mat);
    cur_right_pos = inv_y_roll_mat.multiplyVector4(right_pos);

    // Calculate the normal transformation matrix and pass it to u_normal_mat
    g_normal_mat.setInverseOf(g_model_mat);
    g_normal_mat.transpose();
    gl.uniformMatrix4fv(program.uniforms.u_normal_mat, false, g_normal_mat.elements);

    // Calculate the model view project matrix and pass it to u_mvp_mat
    g_mvp_mat.set(vp_mat);
    g_mvp_mat.multiply(g_model_mat);
    gl.uniformMatrix4fv(program.uniforms.u_mvp_mat, false, g_mvp_mat.elements);
    gl.uniformMatrix4fv(program.uniforms.u_model_mat, false, g_model_mat.elements);
 
    //console.log("g_texture.length = " + g_texture.length);
    if (g_texture.length > 0) {
        // Pass the texure unit to u_image
        gl.uniform1i(program.uniforms.u_image, 0);
        gl.uniform1i(program.uniforms.u_is_texture, true);
    }   

    gl.drawElements(gl.TRIANGLES, g_data.indices.length, gl.UNSIGNED_INT, 0);
}

// call bufferData(...) to send vertices, normals, colors, indices to GPU 
function send_buffer_data (buffer_objects) {
    
    // Write date into the buffer object
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer_objects.vertex_buffer);
    gl.bufferData(gl.ARRAY_BUFFER, g_data.vertices, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffer_objects.normal_buffer);
    gl.bufferData(gl.ARRAY_BUFFER, g_data.normals, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffer_objects.texcoord_buffer);
    gl.bufferData(gl.ARRAY_BUFFER, g_data.texcoords, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffer_objects.barycoord_buffer);
    gl.bufferData(gl.ARRAY_BUFFER, g_data.barycoords, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffer_objects.color_buffer);
    gl.bufferData(gl.ARRAY_BUFFER, g_data.colors, gl.STATIC_DRAW);

    // Write the indices to the buffer object
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer_objects.index_buffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, g_data.indices, gl.STATIC_DRAW);
}

////////////////////////////////////////////////////////////////////////////
function cg_init_framebuffers() {
    console.log("config.IMAGE_X = " + config.IMAGE_X);
    console.log("config.IMAGE_Y = " + config.IMAGE_Y);

    gl.getExtension('EXT_color_buffer_float');
    // enables float framebuffer color attachment

    out_depth = create_double_fbo(config.IMAGE_X, config.IMAGE_Y, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, true);
    out_rainbow = create_double_fbo(config.IMAGE_X, config.IMAGE_Y, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, false);
    outline = create_double_fbo(config.IMAGE_X, config.IMAGE_Y, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, false);
    out = create_double_fbo(config.IMAGE_X, config.IMAGE_Y, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, false);
    gradient = create_double_fbo(config.IMAGE_X, config.IMAGE_Y, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, false);
    mag = create_double_fbo(config.IMAGE_X, config.IMAGE_Y, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, false);
    canny = create_double_fbo(config.IMAGE_X, config.IMAGE_Y, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, false);

	/***** MAJOR EDIT *****/
	// Creating another FBO for rainbow shader
	out_depth_rainbow = create_double_fbo(config.IMAGE_X, config.IMAGE_Y, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, true);
}

// When attaching a texture to a framebuffer, all rendering commands will 
// write to the texture as if it was a normal color/depth or stencil buffer.
// The advantage of using textures is that the result of all rendering operations
// will be stored as a texture image that we can then easily used in shaders
function create_fbo (w, h, internalFormat, format, type, param, depth) {

    //gl.activeTexture(gl.TEXTURE0);
    gl.activeTexture(gl.TEXTURE8); 
    // use high number to avoid confusion with ordinary texture images

    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.MIRRORED_REPEAT);
    // create texture image of resolution (w x h)
    // note that here we pass null as texture source data (no texture image source)
    // For this texture, we're only allocating memory and not actually filling it.
    // Filling texture will happen as soon as we render to the framebuffer.    
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    // make created fbo our main framebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    // attach texture to framebuffer so from now on, everything will be 
    // drawn on this texture image    
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    
	// create a depth renderbuffer
	let depth_buffer = gl.createRenderbuffer();
	gl.bindRenderbuffer(gl.RENDERBUFFER, depth_buffer);

    if (depth) {
		// make a depth buffer and the same size as the targetTexture
		gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
		gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth_buffer);
    }
    
    // if you want to render your whole screen to a texture of a smaller or larger size
    // than the screen, you need to call glViewport again 
    // (before rendering to your framebuffer) with the new dimensions of your texture, 
    // otherwise only a small part of the texture or screen would be drawn onto the texture
    gl.viewport(0, 0, w, h);
    // because framebuffer dimension has changed
    gl.clear(gl.COLOR_BUFFER_BIT);

    let texel_x = 1.0 / w;
    let texel_y = 1.0 / h;

    return {
        texture,
        fbo,
        depth_buffer,
        single: true, // single fbo
        width: w,
        height: h,
        texel_x,
        texel_y,
        internalFormat,
        format,
        type,
        attach(id) {
            gl.activeTexture(gl.TEXTURE0 + id);
            // gl.TEXTURE0, gl.TEXTURE1, ...
            gl.bindTexture(gl.TEXTURE_2D, texture);
            // gl.TEXTURE_2D is now filled by this texture
            return id;
        },
        addTexture(pixel) {
			gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);// do not flip the image's y-axis
			gl.bindTexture(gl.TEXTURE_2D, texture); // fill TEXTURE_2D slot with this FBO's texture 
			gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, gl.FLOAT, pixel);
        }
    };
}

// create 2 FBOs so one pixel processing can be done in-place
function create_double_fbo (w, h, internalFormat, format, type, param, depth) {
    let fbo1 = create_fbo(w, h, internalFormat, format, type, param, depth);
    let fbo2 = create_fbo(w, h, internalFormat, format, type, param, depth);

    let texel_x = 1.0 / w;
    let texel_y = 1.0 / h;

    return {
        width: w,
        height: h,
        single: false, // double fbo
        texel_x,
        texel_y,
        get read() {
            // getter for fbo1
            return fbo1;
        },
        set read(value) {
            fbo1 = value;
        },
        get write() {
            // getter for fbo2
            return fbo2;
        },
        set write(value) {
            fbo2 = value;
        },
        swap() {
            let temp = fbo1;
            fbo1 = fbo2;
            fbo2 = temp;
        }
    }
}

// using glsl 300
function gauss (dst, half) {
    let program = prog_gauss;
    program.bind();
    // drawProgram is now current vertex/fragment shader pair

    if (dst.single) gl.uniform1i(program.uniforms.u_image, dst.attach(8));
    else gl.uniform1i(program.uniforms.u_image, dst.read.attach(8));
    
    gl.uniform2f(program.uniforms.u_texel, dst.texel_x, dst.texel_y);
    gl.uniform1f(program.uniforms.u_half, half);

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    if (dst.single) draw_vao_image(dst.fbo);
    else {
        draw_vao_image(dst.write.fbo);
        dst.swap();
    }
}

function sobel (src, dst, scale) {
    let program = prog_sobel;
    program.bind();

    gl.uniform1i(program.uniforms.u_image, src.read.attach(1));
    gl.uniform2f(program.uniforms.u_texel, src.texel_x, src.texel_y);
    gl.uniform1f(program.uniforms.u_scale, scale);

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    draw_vao_image(dst.write.fbo);
   
    dst.swap();
}

// using glsl 300
function gray (dst) {
    let program = prog_gray;
    program.bind();
    // drawProgram is now current vertex/fragment shader pair

    if (dst.single) gl.uniform1i(program.uniforms.u_image, dst.attach(8));
    else gl.uniform1i(program.uniforms.u_image, dst.read.attach(8));
    
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    if (dst.single) draw_vao_image(dst.fbo);
    else {
        draw_vao_image(dst.write.fbo);
        dst.swap();
    }
}

// use vao to avoid resending data to gpu in each frame
function render_img (src, dst) {
    let program = prog_display;
    program.bind();

    if (src.single) gl.uniform1i(program.uniforms.u_image, src.attach(8));
    else gl.uniform1i(program.uniforms.u_image, src.read.attach(8));
    
    //gl.viewport(0, 0, src.width, src.height);
    gl.viewport(0, 0, dst.width, dst.height);
 
    if (dst.single) draw_vao_image(dst.fbo);
    else {
        draw_vao_image(dst.write.fbo);
        dst.swap();
    }  
}

// render to default framebuffer
function render_null (src) {
    let program = prog_display;
    program.bind();

    if (src.single) gl.uniform1i(program.uniforms.u_image, src.attach(8));
    else gl.uniform1i(program.uniforms.u_image, src.read.attach(8));
    
    gl.viewport(0, 0, canvas.width, canvas.height);

    draw_vao_image(null);
}

function draw_vao_image (fbo) {
    // bind destination fbo to gl.FRAMEBUFFER
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

    // start recording bindBuffer or vertexAttribPointer
  	gl.bindVertexArray(vao_image);
    
    // draw trangles using 6 indices
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

    gl.bindVertexArray(null); // unbind
}

function vao_image_create () {
	// create vao for 2 triangles 
    vao_image = gl.createVertexArray();
    // start recording bindBuffer or vertexAttribPointer
  	gl.bindVertexArray(vao_image);

    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    // we have 4 vertices, forming a 2x2 square
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    // 0 is a reference to attribute variable 'a_position' in shader

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    // note that we have 6 indices in total (3 for each triangle, or half of square)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    // 2 means (x, y)
    
    gl.bindVertexArray(null); // stop recording
}

// read framebuffer content from target.fbo then put it in 1d texture array
function framebufferToTexture (target) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    let length = target.width * target.height * 4;
    // texture is one dimensional array to hold captured image data in CPU memory
    let texture = new Float32Array(length);

    // capture framebuffer (screen) and move image data into CPU memory (texture)
    // gl.readPixels always reads from currently bound framebuffer
    gl.readPixels(0, 0, target.width, target.height, gl.RGBA, gl.FLOAT, texture);
    //gl.readPixels(0, 0, target.width, target.height, gl.RGBA, gl.HALF_FLOAT, texture);
    return texture;
}

// from gradient image, extract gradient magnitude only
function gradient_2_mag (gradient, mag) {
    let program = prog_gradient_2_mag;
    program.bind();

    gl.uniform1i(program.uniforms.u_gradient, gradient.read.attach(1)); 
    
    gl.viewport(0, 0, mag.width, mag.height);

    draw_vao_image(mag.write.fbo);
 
    mag.swap();
}

function nonmaxima_suppression(gradient, mag, canny, thres) {
    let program = prog_nonmaxima_suppression;
    program.bind();

    gl.uniform1i(program.uniforms.u_gradient, gradient.read.attach(1));
    gl.uniform1i(program.uniforms.u_mag, mag.read.attach(2));
    gl.uniform2f(program.uniforms.u_texel, mag.texel_x, mag.texel_y);
    gl.uniform1f(program.uniforms.u_thres, thres);

    gl.viewport(0, 0, canny.width, canny.height);

    if (canny.single)
        draw_vao_image(canny.fbo);
    else {
        draw_vao_image(canny.write.fbo);
        canny.swap();
    }
}

// Note: dilation cannot be applied to single FBO because it will cause 
// Feedback loop formed between Framebuffer and active Texture.
function dilation(src, dst, radius) {
    let program = prog_dilation;
    program.bind();

    if (src.single)
        gl.uniform1i(program.uniforms.u_image, src.attach(1));
    else
        gl.uniform1i(program.uniforms.u_image, src.read.attach(1));

    gl.uniform2f(program.uniforms.u_texel, src.texel_x, src.texel_y);
    gl.uniform1f(program.uniforms.u_radius, radius);

    gl.viewport(0, 0, dst.width, dst.height);

    if (dst.single)
        draw_vao_image(dst.fbo);
    else {
        draw_vao_image(dst.write.fbo);
        dst.swap();
    }
}

// blend double FBO (src) on top of double FBO (dst)
// use src.alpha in each pixel
function blend_src_alpha (src, dst, out) {
    let program = prog_blend_src_alpha;
    program.bind();

    if (src.single) gl.uniform1i(program.uniforms.u_src, src.attach(1));
    else gl.uniform1i(program.uniforms.u_src, src.read.attach(1));

    if (dst.single) gl.uniform1i(program.uniforms.u_dst, dst.attach(2));
    else gl.uniform1i(program.uniforms.u_dst, dst.read.attach(2));

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    if (out.single) draw_vao_image(out.fbo);
    else {
        draw_vao_image(out.write.fbo);
        out.swap();
    }
}

//------------------------------------------------------------------------------
// OBJParser
//------------------------------------------------------------------------------

// OBJ object Constructor
var OBJ = function(obj_filename) {

    this.obj_filename = obj_filename;
    this.mtls = new Array(0);
    // .obj may contain multiple .mtl files
    this.objects = new Array(0);
    // .obj may contain multiple objects
    this.vert = new Array(0);
    // Initialize the vetex coordinates
    this.norm = new Array(0);
    // Initialize the normal vectors
    this.texcoord = new Array(0);
    // Initialize the texture coordinates
}

// Parsing the OBJ file
OBJ.prototype.parse = function(file_string, scale, reverse) {

    var lines = file_string.split('\n');
    // Break up .obj file content into individual lines and store them as array

    lines.push(null);
    // Append null to signal EOF 
    
    var index = 0;
    // current index of array lines
    
    var cur_material_name = ""; // in case there is a material name
    var cur_texture_name = ""; // in case there is a texture image name
    var cur_object = null;    

    // Parse one line at a time
    var line;
    // current line (string)
    
    var sp = new StringParser(); // StringParser is user-defined object
    // StringParser stores current line 

    while ( (line = lines[index++]) != null ) { // parse one line at a time

        sp.init(line);
        // copies current line into StringParser sp

        var command = sp.get_word(); // v, vn, vt, f, ...
        // Get next command (the first word in current line)
                
        if (command == null) continue;
        // null command

        switch (command) {
        case '#': // Skip comments
            continue;            
        case 'mtllib': // Read .mtl file 
            var path = this.parse_mtllib(sp, this.obj_filename);
            // obtain .mtl file name using .obj path
            // path = /home/cs6410/models/cube.mtl
            var mtl = new MTL();
            // Create MTL instance
            this.mtls.push(mtl);
            // .obj file may contain multiple .mtl files 
            
            var mtl_request = new XMLHttpRequest(); // Ajax request
            mtl_request.onreadystatechange = function() {
                if (mtl_request.readyState == 4) { // process complete
                    if (mtl_request.status != 404) { // file found
                        // .mtl file access successful
                        mtl.parse_mtl_file(mtl_request.responseText);
                        // mtl_request.responseText contains .mtl file content as string
                    } 
                    else mtl.ready = true; // .mtl file not found
                }
            }
            mtl_request.open('GET', path, true);
            // Create a request to acquire .mtl file
            // path = /home/cs6410/models/cube.mtl
            mtl_request.send();
            // Send the Ajax request
            continue;
            // Go to the next line
        case 'o': // object name
        case 'g': // group name (treated the same way as 'o')
            // read object name or group namee
            var object = this.parse_object_name(sp);
            // .obj may contain multiple objects 
            this.objects.push(object); // add new object (or group) into array
            cur_object = object;
            continue;
            // Go to the next line
        case 'v':
            if (this.objects.length == 0) { // no registerd object yet
                let object = new Group("default");
                this.objects.push(object); // add new object into array
                cur_object = object;
            }   
            // Read vertex
            var vertex = this.parse_vertex(sp, scale);
            // scale object size so it would fit the canvas
            this.vert.push(vertex); // add vertex to vert array
            continue;
            // Go to the next line
        case 'vn':
            // Read vertex normal
            var normal = this.parse_normal(sp);
            this.norm.push(normal); // add vertex normal to norm array
            continue;
            // Go to the next line
        case 'vt':
            // Read vertex normal
            var texcoord = this.parse_texcoord(sp);
            this.texcoord.push(texcoord); // add texcoord to texcoord array
            continue;
            // Go to the next line
        case 'usemtl': // followed by material name 
            // Read Material name (e.g., "shinyred")
            cur_material_name = this.parse_usemtl(sp);
            continue;
            // Go to the next line
        case 'f':
            // Read face
            var face = this.parse_face(sp, cur_material_name, this.vert, reverse);
            // reverse flips normal vector 
            cur_object.add_face(face);
            continue;
            // Go to the next line
        }
    }

    return true;
}

// get .mtl file path
OBJ.prototype.parse_mtllib = function(sp, obj_filename) {

    // Get directory path of obj_filename
    // e.g., /home/cs6410/models/cube.obj
    var i = obj_filename.lastIndexOf("/"); // locate the last / in the path
    
    let dir_path = "";
    if (i > 0) dir_path = obj_filename.substr(0, i + 1); // /home/cs6410/models/
    
    let mtl_filename = sp.get_word(); // shinyred.mtl 
    console.log(mtl_filename);

    return dir_path + mtl_filename; // /home/cs6410/models/shinyred.mtl
    // Get path
}

// get object name
OBJ.prototype.parse_object_name = function(sp) {
    var object_name = sp.get_word(); // get the next word in file (object name)
    return (new Group(object_name));
}

// get x, y, z for v
OBJ.prototype.parse_vertex = function(sp, scale) {
    var x = sp.get_float() * scale; // scale in x
    var y = sp.get_float() * scale; // scale in y
    var z = sp.get_float() * scale; // scale in z
    return (new Vertex(x,y,z));
}

// get x, y, z for vn
OBJ.prototype.parse_normal = function(sp) {
    var x = sp.get_float();
    var y = sp.get_float();
    var z = sp.get_float();
    return (new Normal(x,y,z));
}

// get x, y for vt
OBJ.prototype.parse_texcoord = function(sp) {
    var x = sp.get_float();
    var y = sp.get_float();
    return (new Texcoord(x,y));
}

// get material name
OBJ.prototype.parse_usemtl = function(sp) {
    return sp.get_word(); // get material name (e.g., "Shinyred")
}

// process 'f 1/1/1 2/2/2 3/3/3 ...'
OBJ.prototype.parse_face = function(sp, material_name, vert, reverse) {
    
    var face = new Face(material_name);

    // process each line of 'f 1/1/1 2/2/2 3/3/3 ...'
    while (true) {  // f 1/1/1 2/2/2 3/3/3 ...
        var word = sp.get_word(); // 1/1/1
        if (word == null) break; // reached end of line so exit loop 
        
        let sub_words = "";
        if (word.search("//") != -1) { // f 1//1 2//2 3//3
            //console.log("// detected!");
            sub_words = word.split('//'); // 1//1

            var vi = parseInt(sub_words[0]) - 1; // vertex index
            face.v_index.push(vi);

            var ni = parseInt(sub_words[1]) - 1; // normal index 
            face.n_index.push(ni);

            face.t_index.push(-1); // no texture index found
        }
        else if (word.search("/") != -1) { // f 1/1/1 2/2/2 3/3/3
            sub_words = word.split('/'); // 1/1/1

            var vi = parseInt(sub_words[0]) - 1; // vertex index
            face.v_index.push(vi);

            var ti = parseInt(sub_words[1]) - 1; // texture index 
            face.t_index.push(ti);

            var ni = parseInt(sub_words[2]) - 1; // normal index 
            face.n_index.push(ni);
        }  
        else { // f 1 2 3 4
            var vi = parseInt(word) - 1; // vertex index
            face.v_index.push(vi);
            face.n_index.push(-1); // no normal index found
            face.t_index.push(-1); // no texture index found
        } 
    }

    // calc face normal, in case vertex normals not available
    // assuming this face is a triangle 
    var v0 = [vert[face.v_index[0]].x, vert[face.v_index[0]].y, vert[face.v_index[0]].z];
    var v1 = [vert[face.v_index[1]].x, vert[face.v_index[1]].y, vert[face.v_index[1]].z];
    var v2 = [vert[face.v_index[2]].x, vert[face.v_index[2]].y, vert[face.v_index[2]].z];
    
    var face_normal = calc_normal(v0, v1, v2); // compute face normal 

    if (face_normal == null) { // normal calculation not possible
        if (face.v_index.length >= 4) { // more complex polygon than triangle 
            var v3 = [vert[face.v_index[3]].x, vert[face.v_index[3]].y, vert[face.v_index[3]].z];
            face_normal = calc_normal(v1, v2, v3);
        }
        if (face_normal == null) { // normal calculation still not possible            
            face_normal = [0.0, 1.0, 0.0];
        }
    }

    if (reverse) { // flip face normal vector 
        face_normal[0] = -face_normal[0];
        face_normal[1] = -face_normal[1];
        face_normal[2] = -face_normal[2];
    }
    
    face.normal = new Normal(face_normal[0], face_normal[1], face_normal[2]);
    // adding to this instance of Face, not the template of Face

    // Divide to triangles if face contains more than 3 vertices
    // this is necessary because we draw gl.TRIANGLES
    if (face.v_index.length > 3) {
        var n2 = face.v_index.length - 2; // n-2 inner triangles within n-gon
        var new_v_index = new Array(n2 * 3);
        var new_n_index = new Array(n2 * 3);
        var new_t_index = new Array(n2 * 3);

        for (var i = 0; i < n2; i++) {
            new_v_index[i * 3 + 0] = face.v_index[0]; // shadred by all inner triangles
            new_v_index[i * 3 + 1] = face.v_index[i + 1];
            new_v_index[i * 3 + 2] = face.v_index[i + 2];
            new_n_index[i * 3 + 0] = face.n_index[0]; // shadred by all inner triangles
            new_n_index[i * 3 + 1] = face.n_index[i + 1];
            new_n_index[i * 3 + 2] = face.n_index[i + 2];
            new_t_index[i * 3 + 0] = face.t_index[0]; // shadred by all inner triangles
            new_t_index[i * 3 + 1] = face.t_index[i + 1];
            new_t_index[i * 3 + 2] = face.t_index[i + 2];
        }
        face.v_index = new_v_index; // this face now has more vertices
        face.n_index = new_n_index; // this face now has more normals
        face.t_index = new_t_index; // this face now has more normals
    }

    face.num_index = face.v_index.length; // this face has this many indices
    // adding to this instance of Face, not the template of Face

    return face;
}

// Check if every material is ready
OBJ.prototype.mtl_ready = function() {

    if (this.mtls.length == 0) return true;

    for (var i = 0; i < this.mtls.length; i++) {        
        if (!this.mtls[i].ready) // found a material not ready
            return false;
    }

    return true;
}

// Check if every texture image is ready
OBJ.prototype.tex_ready = function() {

    for (var i = 0; i < g_texture.length; i++) {       
        if (!g_texture[i].ready) // found a texture not ready
            return false;
    }

    return true;
}

// Find color by material name
OBJ.prototype.find_color = function(material_name) {

    for (var i = 0; i < this.mtls.length; i++) {
        for (var j = 0; j < this.mtls[i].materials.length; j++) {

            // trim() must be used to remove any whitespace chars (even hidden ones)
            if (this.mtls[i].materials[j].name.trim() == material_name.trim()) {
                return (this.mtls[i].materials[j].color);
            }
        }
    }
    return (new Col(0.0, 0.8, 0.0, 1));
}

//------------------------------------------------------------------------------
// Retrieve the information for drawing 3D model
// Create vertices, normals, colors, indices
OBJ.prototype.get_data = function() {
    // Create arrays for vertex coordinates, normals, colors, and indices
    var num_index = 0;

    for (var i = 0; i < this.objects.length; i++) {
        // note that .obj file may contain multiple objects
        num_index += this.objects[i].num_index; // add to the total number of indices
    }
    console.log("num_index = " + num_index);

    var vertices = new Float32Array(num_index * 3); // total number of vertices
    var normals = new Float32Array(num_index * 3); // same number as vertices
    var barycoords = new Float32Array(num_index * 3); // same number as vertices
    var texcoords = new Float32Array(num_index * 2); // same number as vertices
    var colors = new Float32Array(num_index * 4);
    //var indices = new Uint16Array(num_index); // max index: 65,536
    var indices = new Uint32Array(num_index); // total number of indices

    // Set vertex, normal and color
    var ii = 0; // start counting index 

    // OBJ.vertices contains the number of unique vertices
    // vertices contains duplicated vertices shared by multiple faces
    // this is because each vertex may have multiple associated normals
    // in the end, the number of vertices and number of normals must equal

    for (var i = 0; i < this.objects.length; i++) {
        // .obj file may contain multiple objects

        var object = this.objects[i];

        for (var j = 0; j < object.faces.length; j++) { // visit each face

            var face = object.faces[j];

            //console.log("face.material_name = " + face.material_name);
            var color = this.find_color(face.material_name);
            // find material color using material name 
            
            var face_normal = face.normal; // calculated face normal
            // use face_normal in case vn is missing from .obj file            

            for (var k = 0; k < face.v_index.length; k++) {

                // face.v_index.length may be bigger than 3
                // k denotes each vertex in current face

                // Set index
                indices[ii] = ii;
                
                // Copy vertex
                var vIdx = face.v_index[k];
                
                var vertex = this.vert[vIdx];

                vertices[ii * 3 + 0] = vertex.x;
                vertices[ii * 3 + 1] = vertex.y;
                vertices[ii * 3 + 2] = vertex.z;

                barycoords[ii * 3 + 0] = (k % 3 == 0) ? 1 : 0;
                barycoords[ii * 3 + 1] = (k % 3 == 1) ? 1 : 0;
                barycoords[ii * 3 + 2] = (k % 3 == 2) ? 1 : 0;

                var tIdx = face.t_index[k];

                if (tIdx >= 0) { // texcoord exists
                    var texcoord = this.texcoord[tIdx];

                    texcoords[ii * 2 + 0] = texcoord.x;
                    texcoords[ii * 2 + 1] = texcoord.y;
                }
                else {
                    texcoords[ii * 2 + 0] = 0; // default texcoord: (0, 0)
                    texcoords[ii * 2 + 1] = 0; // default texcoord: (0, 0)
                }
                
                // Copy color
                colors[ii * 4 + 0] = color.r;
                colors[ii * 4 + 1] = color.g;
                colors[ii * 4 + 2] = color.b;
                colors[ii * 4 + 3] = color.a;
          
                var nIdx = face.n_index[k];

                // Copy normal
                if (nIdx >= 0) {
                    var normal = this.norm[nIdx]; 

                    normals[ii * 3 + 0] = normal.x;
                    normals[ii * 3 + 1] = normal.y;
                    normals[ii * 3 + 2] = normal.z;
                } 
                else { // use face normals instead
                    normals[ii * 3 + 0] = face_normal.x;
                    normals[ii * 3 + 1] = face_normal.y;
                    normals[ii * 3 + 2] = face_normal.z;
                }

                ii++;
            }
        }
    }
    document.getElementById("info").innerHTML = "vertex count: " + this.vert.length;

    return new DrawingInfo(vertices, normals, texcoords, barycoords, colors, indices);
}

//------------------------------------------------------------------------------
// MTL Object
//------------------------------------------------------------------------------
var MTL = function() {
    this.ready = false;
    // true means MTL is configured correctly
    this.materials = new Array(0);
}

MTL.prototype.parse_newmtl = function(sp) {

    let word = sp.get_word(); // get material name
    //console.log("word = " + word); // 
    
    return word;
}

MTL.prototype.parse_rgb = function(sp, name) {
    // name: material name
    var r = sp.get_float();
    var g = sp.get_float();
    var b = sp.get_float();
    //console.log("[r, g, b] = " + r + " " + g + " " + b);

    return (new Material(name, r, g, b, 1));
}

// get .png file path
MTL.prototype.parse_map_Kd = function(sp) {
 
    let tex_filename = sp.get_word(); // image.png 
    
    tex_filename = url_prefix + tex_filename;
    //console.log(tex_filename);

    return tex_filename; // http://www.cs.umsl.edu/~kang/htdocs/models/image.png
    // Get path
}

// Analyze .mtl file
MTL.prototype.parse_mtl_file = function(file_string) {

    var lines = file_string.split('\n');
    // Break up into lines and store them as array
    lines.push(null);
    // Append null to signal EOF
    var index = 0;
    // Initialize index of line

    // Parse line by line
    var line;
    // A string in the line to be parsed
    var mtl_name = "";
    // Material name
    var sp = new StringParser();
    // Create StringParser

    while ( (line = lines[index++]) != null ) {

        sp.init(line);
        // init StringParser
        var command = sp.get_word();
        // Get command
        if (command == null) continue;
        // null command

        switch (command) {
        case '#':
            continue;
            // Skip comments
        case 'newmtl':
            // Read material name            
            mtl_name = this.parse_newmtl(sp);
            // Get material name
            continue;
            // Go to the next line
        case 'Kd':
            // Read diffuse material color 
            if (mtl_name == "") continue;
            // Go to the next line because material name is unknown
            var material = this.parse_rgb(sp, mtl_name);
            this.materials.push(material);
            mtl_name = "";
            continue;
            // Go to the next line
        case 'map_Kd': // Read .png file for diffuse material color 
            var url = this.parse_map_Kd(sp);
            // url = "http://www.cs.umsl.edu/~kang/htdocs/models/image.png"
            console.log(url);
            load_texture_image(url);
            continue;
            // Go to the next line
        }
    }
    this.ready = true;
}

let load_texture_image = function(url) {

    // Create a texture object
    g_texture.push(gl.createTexture()); 
    let i = g_texture.length - 1;

    g_image.push(new Image());
    g_image[i].crossOrigin = "";
    g_image[i].src = url; 

    g_texture[i].ready = false; // texture not ready yet

    g_image[i].onload = function() { 
        texture_setup(i);
        g_texture[i].ready = true; // texture ready now
    }
}

let texture_setup = function(i) {

  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1); // Flip the image's y axis
  gl.activeTexture(gl.TEXTURE0 + i);
  gl.bindTexture(gl.TEXTURE_2D, g_texture[i]);

  // Set the parameters so we can render any size image.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  // Upload the image into the texture.
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, g_image[i]);
}

//------------------------------------------------------------------------------
// Material Object
//------------------------------------------------------------------------------
var Material = function(name, r, g, b, a) {
    this.name = name;
    this.color = new Col(r, g, b, a);
}

//------------------------------------------------------------------------------
// Vertex Object
//------------------------------------------------------------------------------
var Vertex = function(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
}

//------------------------------------------------------------------------------
// Normal Object
//------------------------------------------------------------------------------
var Normal = function(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
}

//------------------------------------------------------------------------------
// Textcoord Object
//------------------------------------------------------------------------------
var Texcoord = function(x, y) {
    this.x = x;
    this.y = y;
}

//------------------------------------------------------------------------------
// Col Object
//------------------------------------------------------------------------------
var Col = function(r, g, b, a) {
    this.r = r;
    this.g = g;
    this.b = b;
    this.a = a;
}

//------------------------------------------------------------------------------
// Group Object
//------------------------------------------------------------------------------
var Group = function(name) {
    this.name = name;
    this.faces = new Array(0);
    this.num_index = 0;
}

Group.prototype.add_face = function(face) {
    this.faces.push(face);
    this.num_index += face.num_index;
}

//------------------------------------------------------------------------------
// Face Object
//------------------------------------------------------------------------------
// Constructor
var Face = function(material_name) {
    this.material_name = material_name;
    if (material_name == null) this.material_name = "";
    this.v_index = new Array(0);
    this.t_index = new Array(0);
    this.n_index = new Array(0);
}

//------------------------------------------------------------------------------
// DrawInfo Object
//------------------------------------------------------------------------------
// Constructor
var DrawingInfo = function(vertices, normals, texcoords, barycoords, colors, indices) {
    this.vertices = vertices;
    this.normals = normals;
    this.texcoords = texcoords;
    this.barycoords = barycoords;
    this.colors = colors;
    this.indices = indices;
    console.log("indices.length = " + indices.length);
}

//------------------------------------------------------------------------------
// Constructor for StringParser object
var StringParser = function(str) {
    this.str;
    // Store the string specified by the argument
    this.cur;
    // current position in the string to be processed
    this.init(str);
}

// Initialize StringParser object
StringParser.prototype.init = function(str) {
    this.str = str;
    this.cur = 0;
}

// Skip delimiters
StringParser.prototype.skip_delimiters = function() {

    for (var i = this.cur, len = this.str.length; i < len; i++) {
        var c = this.str.charAt(i);
        // Skip TAB, Space, '(', ')
        if (c == '\t' || c == ' ' || c == '(' || c == ')' || c == '"')
            continue;
        break;
    }
    this.cur = i;
}

// Skip to the next word
StringParser.prototype.skip_to_next_word = function() {

    this.skip_delimiters();

    var n = get_word_length(this.str, this.cur);
    this.cur += (n + 1);
}

// Get the next word
StringParser.prototype.get_word = function() {

    this.skip_delimiters(); // get to the next word in the file
     
    // get word length starting from this.cur
    var n = get_word_length(this.str, this.cur);

    if (n == 0) return null;

    var word = this.str.substr(this.cur, n);

    this.cur += (n + 1);

    return word;
}

// Get integer
StringParser.prototype.get_int = function() {

    return parseInt(this.get_word());
}

// Get floating number
StringParser.prototype.get_float = function() {

    return parseFloat(this.get_word());
}

// Get the length of word starting from index start
function get_word_length(str, start) {

    var n = 0;

    for (var i = start, len = str.length; i < len; i++) {
        var c = str.charAt(i);
        if (c == '\t' || c == ' ' || c == '(' || c == ')' || c == '"')
            break;
    }

    return i - start;
}

//------------------------------------------------------------------------------
// Common function
//------------------------------------------------------------------------------
// compute face normal of a triangle 
function calc_normal(p0, p1, p2) {
    // v0: a vector from p1 to p0, v1; a vector from p1 to p2
    var v0 = new Float32Array(3);
    var v1 = new Float32Array(3);
    for (var i = 0; i < 3; i++) {
        v0[i] = p0[i] - p1[i];
        v1[i] = p2[i] - p1[i];
    }

    // The cross product of v0 and v1
    var c = new Float32Array(3);
    c[0] = v0[1] * v1[2] - v0[2] * v1[1];
    c[1] = v0[2] * v1[0] - v0[0] * v1[2];
    c[2] = v0[0] * v1[1] - v0[1] * v1[0];

    // Normalize the result
    var v = new Vector3(c); // defined in cuon-matrix.js
    v.normalize();

    return v.elements; // unpack into JavaScript array
}

// compute camera position 
function calc_camera_pos() {
    // v0: a vector from p1 to p0, v1; a vector from p1 to p2
    let d = new THREE.Vector3(0, 20, 40);
    d = d.normalize();
    d = d.multiplyScalar(config.CAMERA_DIST);

    return d; // unpack into JavaScript array
}

//////////////////////////////////////////////////////////////////////////////////
// mouse controls
////////////////////////////////////////////////////////////////////////////
function cg_register_event_handlers() {
    canvas.addEventListener("wheel", cg_wheel);
    canvas.addEventListener("mousedown", cg_mousedown);
    canvas.addEventListener("mouseup", cg_mouseup);
    canvas.addEventListener("mousemove", cg_mousemove);
}

function cg_wheel (e) {
	if (e.deltaY > 0) { // going down (zoom out)
	   config.CAMERA_DIST += 5.0;
	   render();
	}
	else { // going up (zoom in)
	   config.CAMERA_DIST -= 5.0;
	   if (config.CAMERA_DIST < 5) config.CAMERA_DIST = 5.0;
	   render();
	}
}

let dragging = false;         // Dragging or not
let old_x = -1, old_y = -1;   // Last position of the mouse

function cg_mousedown (e) {
    console.log("pressed");
    // (x, y): mouse position within canvas    
    var x = e.offsetX, y = e.offsetY;
    // Start dragging 
    old_x = x; old_y = y;
    dragging = true;    
};

function cg_mouseup (e) {
    dragging = false;
};

function cg_mousemove (e) {
    var x = e.offsetX, y = e.offsetY;
    if (dragging) {
      var factor = 0.2; // Rotation factor
      var dx = factor * (x - old_x); // how much horizontal move (y-roll)
      var dy = factor * (y - old_y); // how much vertical move (x-roll)
      config.SPEED_X += dy;
      config.SPEED_Y += dx;
    }
    old_x = x, old_y = y;
};

////////////////////////////////////////////////////////////////////////////////////////
