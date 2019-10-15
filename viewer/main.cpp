#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <SDL.h>
#include "app_util.h"
#include "arcball_camera.h"
#include "ba_tree.h"
#include "borrowed_array.h"
#include "debug.h"
#include "file_mapping.h"
#include "gl_core_4_5.h"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl.h"
#include "lba_tree_builder.h"
#include "shader.h"
#include "transfer_function_widget.h"
#include "util.h"
#include <glm/gtx/color_space.hpp>

const std::string USAGE = "Usage: ./viewer [file.xyz...]";

struct TreeData {
    GLuint points_vbo = -1;
    GLuint planes_vbo = -1;
    GLuint attrib_vbo = -1;

    std::vector<glm::vec3> query_pts;
    std::vector<glm::vec3> plane_verts;
    std::vector<std::string> attrib_names;
    std::vector<AttributeQuery> attribute_queries;

    int selected_attrib = 0;
    glm::vec2 query_range = glm::vec2(0);
    int shader = 0;
    bool display = true;
    std::string file;
};

int win_width = 1280;
int win_height = 720;

BATree load_xyz(const std::string &fname);

void run_app(const std::vector<std::string> &args, SDL_Window *window);

glm::vec2 transform_mouse(glm::vec2 in)
{
    return glm::vec2(in.x * 2.f / win_width - 1.f, 1.f - 2.f * in.y / win_height);
}

std::vector<glm::vec3> make_box_verts(const Box &b);
std::vector<glm::vec3> make_plane_verts(const Plane &p);

int main(int argc, char **argv)
{
    const std::vector<std::string> args(argv, argv + argc);
    if (args.size() == 1) {
        std::cout << USAGE << "\n";
        return 1;
    }

    if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        std::cerr << "Failed to init SDL: " << SDL_GetError() << "\n";
        return -1;
    }

    const char *glsl_version = "#version 450 core";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
#ifdef DEBUG_GL
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_Window *window = SDL_CreateWindow("XYZ Viewer",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          win_width,
                                          win_height,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_SetSwapInterval(1);
    SDL_GL_MakeCurrent(window, gl_context);

    if (ogl_LoadFunctions() == ogl_LOAD_FAILED) {
        std::cerr << "Failed to initialize OpenGL\n";
        return 1;
    }
#ifdef DEBUG_GL
    register_debug_callback();
#endif

    // Setup Dear ImGui context
    ImGui::CreateContext();

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    run_app(args, window);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

void run_app(const std::vector<std::string> &args, SDL_Window *window)
{
    ImGuiIO &io = ImGui::GetIO();

    const std::string base_path = get_base_path();

    ArcballCamera camera(glm::vec3(0, 0, 5), glm::vec3(0), glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(
        glm::radians(65.f), static_cast<float>(win_width) / win_height, 0.1f, 1000.f);

    std::vector<Shader> shaders = {
        Shader(base_path + "shaders/vert.glsl", base_path + "shaders/frag.glsl"),
        Shader(base_path + "shaders/int_attrib_vert.glsl", base_path + "shaders/frag.glsl"),
        Shader(base_path + "shaders/float_attrib_vert.glsl", base_path + "shaders/frag.glsl"),
        Shader(base_path + "shaders/double_attrib_vert.glsl", base_path + "shaders/frag.glsl")};

    Box world_bounds;

    std::vector<BATree> trees;
    std::vector<TreeData> tree_data;
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i][0] == '-') {
            continue;
        }
        trees.push_back(load_xyz(args[i]));

        std::cout << "Tree '" << args[i] << "' info:\n"
                  << "bounds: " << trees.back().tree_bounds << "\n"
                  << "# nodes: " << trees.back().nodes->size() << "\n"
                  << "# points: " << trees.back().points->size() << "\n"
                  << "# attribs: " << trees.back().attribs.size() << "\n";

        TreeData data;
        data.file = args[i];
        world_bounds.box_union(trees.back().tree_bounds);

        for (size_t j = 0; j < trees.back().attribs.size(); ++j) {
            const auto &a = trees.back().attribs[j];
            data.attrib_names.push_back(a.name);
            std::cout << "  attrib: " << a.name << ", "
                      << " # elems: " << a.size() << "\n";
        }

        glCreateBuffers(1, &data.points_vbo);
        glCreateBuffers(1, &data.planes_vbo);
        glCreateBuffers(1, &data.attrib_vbo);
        tree_data.push_back(data);
    }
    std::cout << "Total world bounds: " << world_bounds << "\n";

    Box query_box = world_bounds;
    std::vector<glm::vec3> query_box_verts;

    GLuint vao;
    glCreateVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint query_box_vbo;
    glCreateBuffers(1, &query_box_vbo);

    glEnableVertexAttribArray(0);

    TransferFunctionWidget tfn_widget;
    // Texture for sampling the transfer function data
    GLuint colormap_texture;
    glGenTextures(1, &colormap_texture);
    glBindTexture(GL_TEXTURE_1D, colormap_texture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_LOD, 0);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAX_LOD, 0);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);
    {
        auto colormap = tfn_widget.get_colormap();
        glTexImage1D(GL_TEXTURE_1D,
                     0,
                     GL_RGBA8,
                     colormap.size() / 4,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     colormap.data());
    }

    float point_size = 5.f;
    glPointSize(point_size);
    glClearColor(0.1f, 0.1f, 0.1f, 1.f);
    glClearDepth(1.f);
    glEnable(GL_DEPTH_TEST);

    glm::vec2 prev_mouse(-2.f);
    size_t total_points = 0;
    size_t displayed_points = 0;
    bool done = false;
    bool draw_splitting_planes = false;
    bool draw_query_box = false;
    bool box_changed = true;
    bool attrib_changed = true;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                done = true;
            }
            if (event.type == SDL_WINDOWEVENT) {
                if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                    done = true;
                } else if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    win_width = event.window.data1;
                    win_height = event.window.data2;

                    proj = glm::perspective(glm::radians(65.f),
                                            static_cast<float>(win_width) / win_height,
                                            0.1f,
                                            500.f);
                }
            }
            if (!io.WantCaptureKeyboard) {
                if (event.type == SDL_KEYDOWN) {
                    switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        done = true;
                        break;
                    default:
                        break;
                    }
                }
            }
            if (!io.WantCaptureMouse) {
                if (event.type == SDL_MOUSEMOTION) {
                    const glm::vec2 cur_mouse =
                        transform_mouse(glm::vec2(event.motion.x, event.motion.y));
                    if (prev_mouse != glm::vec2(-2.f)) {
                        if (event.motion.state & SDL_BUTTON_LMASK) {
                            camera.rotate(prev_mouse, cur_mouse);
                        } else if (event.motion.state & SDL_BUTTON_RMASK) {
                            camera.pan(cur_mouse - prev_mouse);
                        }
                    }
                    prev_mouse = cur_mouse;
                } else if (event.type == SDL_MOUSEWHEEL) {
                    camera.zoom(event.wheel.y * 0.1);
                }
            }
        }

        if (tfn_widget.changed()) {
            auto colormap = tfn_widget.get_colormap();
            glTexImage1D(GL_TEXTURE_1D,
                         0,
                         GL_RGBA8,
                         colormap.size() / 4,
                         0,
                         GL_RGBA,
                         GL_UNSIGNED_BYTE,
                         colormap.data());
        }

        const glm::mat4 proj_view = proj * camera.transform();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();

        if (ImGui::Begin("Transfer Function")) {
            tfn_widget.draw_ui();
        }
        ImGui::End();

        ImGui::Begin("Debug Panel");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);

        if (ImGui::SliderFloat("Point Size", &point_size, 1.f, 10.f)) {
            glPointSize(point_size);
        }

        // We need a separate slider for each b/c we have different ranges for each
        std::array<float, 2> x_range = {query_box.lower.x, query_box.upper.x};
        std::array<float, 2> y_range = {query_box.lower.y, query_box.upper.y};
        std::array<float, 2> z_range = {query_box.lower.z, query_box.upper.z};

        const std::string pretty_total = pretty_print_count(total_points);
        const std::string pretty_displayed = pretty_print_count(displayed_points);
        ImGui::Text("Query Box");
        ImGui::Text("Queried Points: %s", pretty_total.c_str());
        ImGui::Text("Displayed Points: %s", pretty_displayed.c_str());
        box_changed |= ImGui::SliderFloat2(
            "x range", x_range.data(), world_bounds.lower.x, world_bounds.upper.x);
        box_changed |= ImGui::SliderFloat2(
            "y range", y_range.data(), world_bounds.lower.y, world_bounds.upper.y);
        box_changed |= ImGui::SliderFloat2(
            "z range", z_range.data(), world_bounds.lower.z, world_bounds.upper.z);

        ImGui::Checkbox("Draw splitting planes", &draw_splitting_planes);
        ImGui::Checkbox("Draw query box", &draw_query_box);

        for (size_t i = 0; i < tree_data.size(); ++i) {
            auto &td = tree_data[i];
            ImGui::PushID(i);

            ImGui::Text("Tree %s", td.file.c_str());
            ImGui::Checkbox("Display", &td.display);

            if (!td.attrib_names.empty()) {
                std::vector<const char *> strs;
                std::transform(td.attrib_names.begin(),
                               td.attrib_names.end(),
                               std::back_inserter(strs),
                               [](const std::string &s) { return s.c_str(); });

                attrib_changed |=
                    ImGui::Combo("Attributes", &td.selected_attrib, strs.data(), strs.size());

                if (attrib_changed) {
                    td.query_range = trees[i].attribs[td.selected_attrib].range;
                }

                attrib_changed |= ImGui::SliderFloat2("Query Range",
                                                      &td.query_range.x,
                                                      trees[i].attribs[td.selected_attrib].range.x,
                                                      trees[i].attribs[td.selected_attrib].range.y);
                if (td.query_range.x > td.query_range.y) {
                    std::swap(td.query_range.x, td.query_range.y);
                }
            }

            ImGui::PopID();
        }

        ImGui::End();

        if (box_changed || attrib_changed) {
            total_points = 0;
            displayed_points = 0;
            box_changed = false;
            attrib_changed = false;

            glm::vec3 lower(x_range[0], y_range[0], z_range[0]);
            glm::vec3 upper(x_range[1], y_range[1], z_range[1]);
            query_box = Box(glm::min(lower, upper), glm::max(lower, upper));

            for (size_t i = 0; i < trees.size(); ++i) {
                using namespace std::chrono;

                const BATree &tree = trees[i];
                TreeData &data = tree_data[i];

                data.query_pts.clear();
                if (!data.attrib_names.empty()) {
                    data.attribute_queries.clear();
                    // TODO: If the attrib queries aren't empty we can re-use the already alloc'd
                    // memory instead of making a new attrib query (and thus new buffer) each time.
                    data.attribute_queries.emplace_back(data.attrib_names[data.selected_attrib],
                                                        data.query_range);
                }

                auto start = high_resolution_clock::now();
                tree.query_box(query_box, data.query_pts, &data.attribute_queries);
                auto end = high_resolution_clock::now();
                auto dur = duration_cast<milliseconds>(end - start);
                std::cout << "Query of " << pretty_print_count(data.query_pts.size()) << " with "
                          << data.attribute_queries.size() << " attrib queries took " << dur.count()
                          << "ms\n";

                total_points += data.query_pts.size();
                if (data.display) {
                    displayed_points += data.query_pts.size();
                }

                if (!data.query_pts.empty()) {
                    glBindBuffer(GL_ARRAY_BUFFER, data.points_vbo);
                    glBufferData(GL_ARRAY_BUFFER,
                                 data.query_pts.size() * sizeof(glm::vec3),
                                 data.query_pts.data(),
                                 GL_STATIC_DRAW);

                    data.shader = 0;
                    if (!data.attribute_queries.empty()) {
                        if (data.attribute_queries[0].data_type == INT_32) {
                            data.shader = 1;
                        } else if (data.attribute_queries[0].data_type == FLOAT_32) {
                            data.shader = 2;
                        } else if (data.attribute_queries[0].data_type == FLOAT_64) {
                            data.shader = 3;
                        } else {
                            std::cout << "Warning: Unhandled attribute/shader type combination\n";
                        }
                        std::cout << "# particles: " << data.query_pts.size() << "\n"
                                  << "attrib type: "
                                  << print_data_type(data.attribute_queries[0].data_type)
                                  << ", # attrib queried: " << data.attribute_queries[0].size()
                                  << ", stride: " << data.attribute_queries[0].stride() << "\n";

                        glBindBuffer(GL_ARRAY_BUFFER, data.attrib_vbo);
                        glBufferData(GL_ARRAY_BUFFER,
                                     data.attribute_queries[0].data->size(),
                                     data.attribute_queries[0].data->data(),
                                     GL_STATIC_DRAW);
                    }

                    // For debug vis: query the splitting planes of the tree down this path
                    std::vector<Plane> splitting_planes;
                    tree.get_splitting_planes(splitting_planes, query_box, &data.attribute_queries);
                    data.plane_verts.clear();
                    for (const auto &p : splitting_planes) {
                        const auto plane_verts = make_plane_verts(p);
                        std::copy(plane_verts.begin(),
                                  plane_verts.end(),
                                  std::back_inserter(data.plane_verts));
                    }
                    glBindBuffer(GL_ARRAY_BUFFER, data.planes_vbo);
                    glBufferData(GL_ARRAY_BUFFER,
                                 data.plane_verts.size() * sizeof(glm::vec3),
                                 data.plane_verts.data(),
                                 GL_STATIC_DRAW);
                }
            }

            query_box_verts = make_box_verts(query_box);
            glBindBuffer(GL_ARRAY_BUFFER, query_box_vbo);
            glBufferData(GL_ARRAY_BUFFER,
                         query_box_verts.size() * sizeof(glm::vec3),
                         query_box_verts.data(),
                         GL_STATIC_DRAW);
        }

        // Rendering
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_1D, colormap_texture);

        for (size_t i = 0; i < tree_data.size(); ++i) {
            const TreeData &td = tree_data[i];
            const BATree &tree = trees[i];

            if (!td.display) {
                continue;
            }

            shaders[td.shader].use();
            shaders[td.shader].uniform("proj_view", proj_view);
            shaders[td.shader].uniform(
                "fcolor", glm::rgbColor(glm::vec3((360.f * i) / tree_data.size(), 1.f, 0.7f)));

            if (td.shader != 0) {
                shaders[td.shader].uniform("range", tree.attribs[td.selected_attrib].range);
            }

            glBindBuffer(GL_ARRAY_BUFFER, td.points_vbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

            glBindBuffer(GL_ARRAY_BUFFER, td.attrib_vbo);
            if (!td.attribute_queries.empty()) {
                glEnableVertexAttribArray(1);
                if (td.attribute_queries[0].data_type == INT_32) {
                    glVertexAttribIPointer(1, 1, GL_INT, 0, 0);
                } else if (td.attribute_queries[0].data_type == FLOAT_32) {
                    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
                } else if (td.attribute_queries[0].data_type == FLOAT_64) {
                    glVertexAttribLPointer(1, 1, GL_DOUBLE, 0, 0);
                }
            }
            glDrawArrays(GL_POINTS, 0, td.query_pts.size());
            glDisableVertexAttribArray(1);

            if (draw_splitting_planes) {
                glBindBuffer(GL_ARRAY_BUFFER, td.planes_vbo);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

                shaders[0].use();
                for (size_t j = 0; j < td.plane_verts.size() / 4; ++j) {
                    shaders[0].uniform("proj_view", proj_view);
                    shaders[0].uniform("fcolor",
                                       glm::rgbColor(glm::vec3(
                                           (4.f * 360.f * j) / td.plane_verts.size(), 0.7f, 0.7f)));
                    glDrawArrays(GL_LINE_LOOP, j * 4, 4);
                }
            }
        }
        if (draw_query_box) {
            glBindBuffer(GL_ARRAY_BUFFER, query_box_vbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

            shaders[0].use();
            shaders[0].uniform("proj_view", proj_view);
            shaders[0].uniform("fcolor", glm::vec3(1.f, 0.75f, 0.f));
            glDrawArrays(GL_LINES, 0, query_box_verts.size());
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);
    }
}

std::vector<glm::vec3> make_box_verts(const Box &b)
{
    return std::vector<glm::vec3>{b.lower,
                                  glm::vec3(b.upper.x, b.lower.y, b.lower.z),

                                  glm::vec3(b.upper.x, b.lower.y, b.lower.z),
                                  glm::vec3(b.upper.x, b.upper.y, b.lower.z),

                                  glm::vec3(b.upper.x, b.upper.y, b.lower.z),
                                  glm::vec3(b.lower.x, b.upper.y, b.lower.z),

                                  glm::vec3(b.lower.x, b.upper.y, b.lower.z),
                                  b.lower,

                                  glm::vec3(b.lower.x, b.lower.y, b.upper.z),
                                  glm::vec3(b.upper.x, b.lower.y, b.upper.z),

                                  glm::vec3(b.upper.x, b.lower.y, b.upper.z),
                                  glm::vec3(b.upper.x, b.upper.y, b.upper.z),

                                  glm::vec3(b.upper.x, b.upper.y, b.upper.z),
                                  glm::vec3(b.lower.x, b.upper.y, b.upper.z),

                                  glm::vec3(b.lower.x, b.upper.y, b.upper.z),
                                  glm::vec3(b.lower.x, b.lower.y, b.upper.z),

                                  b.lower,
                                  glm::vec3(b.lower.x, b.lower.y, b.upper.z),

                                  glm::vec3(b.upper.x, b.lower.y, b.lower.z),
                                  glm::vec3(b.upper.x, b.lower.y, b.upper.z),

                                  glm::vec3(b.upper.x, b.upper.y, b.lower.z),
                                  glm::vec3(b.upper.x, b.upper.y, b.upper.z),

                                  glm::vec3(b.lower.x, b.upper.y, b.lower.z),
                                  glm::vec3(b.lower.x, b.upper.y, b.upper.z)};
}

std::vector<glm::vec3> make_plane_verts(const Plane &p)
{
    glm::vec3 dir = glm::vec3(-1.f);
    if (p.half_vectors.x == 0.f) {
        dir = glm::vec3(0.f, 1.f, -1.f);
    } else if (p.half_vectors.y == 0.f) {
        dir = glm::vec3(1.f, 0.f, -1.f);
    } else {
        dir = glm::vec3(1.f, -1.f, 0.f);
    }

    return std::vector<glm::vec3>{p.origin - p.half_vectors,
                                  p.origin + dir * p.half_vectors,
                                  p.origin + p.half_vectors,
                                  p.origin - dir * p.half_vectors};
}

BATree load_xyz(const std::string &fname)
{
    std::ifstream fin(fname.c_str());

    // First line has the number of atoms, second has the name of the data
    std::string line;
    std::getline(fin, line);
    const size_t num_atoms = std::stoi(line);
    std::cout << "Expecting: " << num_atoms << " atoms\n";

    std::getline(fin, line);
    std::cout << "Molecule name: " << line << "\n";

    // XYZ format is assumed to be TYPE X Y Z
    std::vector<glm::vec3> points;
    std::vector<int> atom_ids;
    int next_atom_id = 0;
    std::unordered_map<std::string, int> atom_id_map;
    while (std::getline(fin, line)) {
        if (line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        std::string type;
        float x, y, z;
        ss >> type >> x >> y >> z;

        int atom_id = -1;
        auto fnd = atom_id_map.find(type);
        if (fnd != atom_id_map.end()) {
            atom_id = fnd->second;
        } else {
            atom_id = next_atom_id++;
            atom_id_map[type] = atom_id;
        }

        points.emplace_back(x, y, z);
        atom_ids.push_back(atom_id);
        if (points.size() == num_atoms) {
            break;
        }
    }

    auto atom_arr = std::make_shared<OwnedArray<uint8_t>>(
        reinterpret_cast<uint8_t *>(atom_ids.data()), sizeof(int) * atom_ids.size());

    std::vector<Attribute> attributes = {Attribute("atom_id", atom_arr, DTYPE::INT_32)};

    return LBATreeBuilder(std::move(points), std::move(attributes)).compact();
}

