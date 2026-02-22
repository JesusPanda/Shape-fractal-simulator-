#include <SFML/Graphics.hpp>
#include "kernels.cuh"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <sstream>

// --- Viewport and Camera ---
struct Camera {
    sf::Vector2f center = {0.5f, 0.5f};
    float zoom = 1.0f;
};

// --- World to Screen Transformation ---
sf::Vector2f to_screen(const float2& world_pos, const Camera& cam, const sf::Vector2u& win_size) {
    float x = (world_pos.x - cam.center.x) * cam.zoom * win_size.x + (win_size.x / 2.0f);
    float y = (world_pos.y - cam.center.y) * cam.zoom * win_size.y + (win_size.y / 2.0f);
    return {x, y};
}

// --- UI Text Helper ---
void draw_text(sf::RenderWindow& window, const std::string& str, const sf::Font& font, int char_size, sf::Vector2f pos, sf::Color color) {
    sf::Text text(str, font, char_size);
    text.setPosition(pos);
    text.setFillColor(color);
    window.draw(text);
}

// --- Font Discovery Helper ---
static std::string find_font_path(int argc, char* argv[]) {
    // 1. Check command line arguments
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--font") {
            return argv[i + 1];
        }
    }

    // 2. Check environment variable
    const char* env_p = std::getenv("TRIGSIM_FONT");
    if (env_p) return std::string(env_p);

    // 3. Local directory
    std::vector<std::string> local_paths = {"DejaVuSans.ttf"};
    for (const auto& p : local_paths) {
        if (std::filesystem::exists(p)) return p;
    }

    // 4. Fallback to original system path
    std::string fallback = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    if (std::filesystem::exists(fallback)) return fallback;

    return "";
}

int main(int argc, char* argv[]) {
    // --- Window and SFML Setup ---
    const unsigned int WIN_WIDTH = 1200, WIN_HEIGHT = 1200;
    sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Recursive Right-Triangle (CUDA C++)", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    sf::Font font;
    // Note: This requires a font file to be available.
    // Use the discovery helper to find a font path.
    std::string font_path = find_font_path(argc, argv);
    if (font_path.empty() || !font.loadFromFile(font_path)) {
        std::cerr << "Warning: Could not load font. HUD will not be displayed." << std::endl;
        if (!font_path.empty()) std::cerr << "Attempted path: " << font_path << std::endl;
    }

    // --- Simulation State ---
    SimData gpu_data;
    Camera camera;
    float angle_deg = 30.0f;
    int depth = 2;
    std::string iter_buffer = "2";

    // --- Cache and Input State ---
    float last_angle_key = -1.f;
    int last_depth = -1;
    bool rmb_dragging = false;
    sf::Vector2i last_mouse_pos;

    // --- Initialize GPU Memory ---
    init_memory(gpu_data);

    // --- Main Loop ---
    while (window.isOpen()) {
        // --- Event Handling ---
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            // Zoom
            if (event.type == sf::Event::MouseWheelScrolled) {
                float zoom_factor = std::pow(1.02f, -event.mouseWheelScroll.delta);
                camera.zoom *= zoom_factor;
                camera.zoom = std::max(0.2f, std::min(camera.zoom, 50.0f));
            }
            // Reset
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::R) {
                camera = Camera();
                angle_deg = 30.0f;
                depth = 2;
                iter_buffer = "2";
            }
            // Iteration buffer
            if (event.type == sf::Event::TextEntered) {
                if (isdigit(event.text.unicode)) {
                    if (!(event.text.unicode == '0' && iter_buffer.empty())) {
                        iter_buffer += static_cast<char>(event.text.unicode);
                        try { depth = std::stoi(iter_buffer); } catch(...) { depth = MAX_ITERS; }
                        depth = std::min(depth, MAX_ITERS);
                    }
                } else if (event.text.unicode == 8 && !iter_buffer.empty()) { // Backspace
                    iter_buffer.pop_back();
                    try { depth = iter_buffer.empty() ? 0 : std::stoi(iter_buffer); } catch(...) { depth = 0; }
                }
            }
        }

        // --- RMB Pan ---
        if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
            sf::Vector2i current_mouse_pos = sf::Mouse::getPosition(window);
            if (rmb_dragging) {
                sf::Vector2f delta = window.mapPixelToCoords(current_mouse_pos) - window.mapPixelToCoords(last_mouse_pos);
                camera.center.x -= delta.x / (camera.zoom * WIN_WIDTH);
                camera.center.y -= delta.y / (camera.zoom * WIN_HEIGHT);
            }
            last_mouse_pos = current_mouse_pos;
            rmb_dragging = true;
        } else {
            rmb_dragging = false;
        }

        // --- Rebuild Geometry on Change ---
        float angle_key = roundf(angle_deg * 100) / 100;
        if (angle_key != last_angle_key || depth != last_depth) {
            reset_and_build_base_launcher(gpu_data, angle_deg);
            int count = 1;
            int current_buf = 0;
            for (int i = 0; i < depth; ++i) {
                expand_once_launcher(gpu_data, current_buf, 1 - current_buf, count);
                count *= 2;
                current_buf = 1 - current_buf;
            }
            last_angle_key = angle_key;
            last_depth = depth;
        }

        // --- Drawing ---
        window.clear(sf::Color::Black);

        // Get counts from GPU
        int seg_c = 0, hyp_c = 0;
        get_counts(gpu_data, seg_c, hyp_c);
        seg_c = std::min(seg_c, (int)SEG_CAP);
        hyp_c = std::min(hyp_c, (int)HYPO_CAP);

        sf::Vector2u win_size = window.getSize();

        // Launch GPU kernel to update visualization buffers
        update_visualization_launcher(gpu_data, make_float2(camera.center.x, camera.center.y), camera.zoom, make_int2(win_size.x, win_size.y), seg_c, hyp_c);

        // Draw directly from GPU/Managed memory
        if (seg_c > 0)
            window.draw(reinterpret_cast<sf::Vertex*>(gpu_data.gl_segments), seg_c * 2, sf::Lines);
        if (hyp_c > 0)
            window.draw(reinterpret_cast<sf::Vertex*>(gpu_data.gl_hypotenuses), hyp_c * 2, sf::Lines);

        // Draw A,B,C markers
        float2 a, b, c;
        get_abc_points(gpu_data, a, b, c);
        sf::CircleShape marker(4.f);
        marker.setOrigin(4.f, 4.f);
        marker.setPosition(to_screen(a, camera, win_size));
        marker.setFillColor(sf::Color(255, 85, 85));
        window.draw(marker);
        marker.setPosition(to_screen(b, camera, win_size));
        marker.setFillColor(sf::Color(85, 255, 85));
        window.draw(marker);
        marker.setPosition(to_screen(c, camera, win_size));
        marker.setFillColor(sf::Color(85, 85, 255));
        window.draw(marker);


        // --- HUD ---
        // This is a simplified HUD. A real implementation might use a GUI library.
        // For now, we manually handle a simple angle slider concept.
        // A simple clickable area for angle adjustment
        sf::RectangleShape angle_bar(sf::Vector2f(200, 5));
        angle_bar.setPosition(20, 50);
        angle_bar.setFillColor(sf::Color(100, 100, 100));
        window.draw(angle_bar);

        sf::CircleShape angle_handle(8);
        angle_handle.setOrigin(8, 8);
        float handle_x = 20.0f + (angle_deg - 5.0f) / (85.0f - 5.0f) * 200.0f;
        angle_handle.setPosition(handle_x, 52.5f);
        angle_handle.setFillColor(sf::Color::White);
        window.draw(angle_handle);

        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
            if (sf::FloatRect(20, 40, 200, 25).contains((float)mouse_pos.x, (float)mouse_pos.y)) {
                angle_deg = 5.0f + ((float)mouse_pos.x - 20.0f) / 200.0f * (85.0f - 5.0f);
                angle_deg = std::max(5.0f, std::min(85.0f, angle_deg));
            }
        }

        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << angle_deg;

        draw_text(window, "Angle A: " + ss.str() + " deg", font, 16, {20, 20}, sf::Color::White);
        draw_text(window, "Iterations: " + std::to_string(depth) + " (cap: " + std::to_string(MAX_ITERS) + ")", font, 16, {20, 70}, sf::Color::White);
        draw_text(window, "Buffer: '" + iter_buffer + "'", font, 16, {20, 95}, sf::Color(128,128,128));
        draw_text(window, "RMB Pan | Wheel Zoom | R Reset", font, 14, {20, 120}, sf::Color(150,150,150));


        window.display();
    }

    // --- Cleanup ---
    free_memory(gpu_data);

    return 0;
}