#include <memory>
#include <string>
#include <vector>

#include <cstdio>

class DescriptiveData {
public:
    const std::string &get_title() const { return title_; }
    void set_title(const std::string &title) { title_ = title; }

private:
    std::string title_;
};

class Variable {
public:
    Variable(const char *name) : name_(name) {}
    const std::string name() const { return name_; }
    void add_value(const std::string &value) { values_.push_back(value); }
    size_t num_scenarios() const { return values_.size(); }
    const std::string &get_value(unsigned scenario) const { return values_[scenario]; }
    double get_value_as_double(unsigned scenario) const {
        auto s = get_value(scenario);
        return strtod(s.c_str(), NULL);
    }

private:
    std::string name_;
    std::vector<std::string> values_;
};

class AsphericalData {
public:
    AsphericalData(int surface_number) : surface_number_(surface_number) {}
    void add_data(double d) { data_.push_back(d); }
    int data_points() const { return data_.size(); }
    double data(int i) const {
        return i >= 0 && i < data_.size() ? data_[i] : 0.0;
    }

private:
    int surface_number_;
    std::vector<double> data_;
};

enum SurfaceType { surface,
                   aperture_stop,
};

class Surface {
public:
    Surface(int id)
        : id_(id), surface_type_(SurfaceType::surface), radius_(0),
          diameter_(0), refractive_index_(0), abbe_vd_(0), is_cover_glass_(false) {}
    SurfaceType get_surface_type() const { return surface_type_; }
    void set_surface_type(SurfaceType surface_type) {
        surface_type_ = surface_type;
    }
    double get_radius() const { return radius_; }
    void set_radius(double radius) { radius_ = radius; }
    double get_thickness(unsigned scenario) const { return thickness_by_scenario_[scenario]; }
    void add_thickness(double thickness) { thickness_by_scenario_.push_back(thickness); }
    double get_diameter() const { return diameter_; }
    void set_diameter(double value) { diameter_ = value; }
    double get_refractive_index() const { return refractive_index_; }
    void set_refractive_index(double refractive_index) {
        refractive_index_ = refractive_index;
    }
    double get_abbe_vd() const { return abbe_vd_; }
    void set_abbe_vd(double abbe_vd) { abbe_vd_ = abbe_vd; }
    std::shared_ptr<AsphericalData> get_aspherical_data() const { return aspherical_data_; }
    void set_aspherical_data(std::shared_ptr<AsphericalData> aspherical_data) {
        aspherical_data_ = aspherical_data;
    }
    int get_id() const { return id_; }
    bool is_cover_glass() const {
        return is_cover_glass_;
    }
    void set_is_cover_glass(bool is_cover_glass) {
        is_cover_glass_ = is_cover_glass;
    }

private:
    int id_;
    SurfaceType surface_type_;
    double radius_;
    std::vector<double> thickness_by_scenario_;
    double diameter_;
    double refractive_index_;
    double abbe_vd_;
    bool is_cover_glass_;
    std::shared_ptr<AsphericalData> aspherical_data_;
};

class LensSystem {
public:
    bool parse_file(const std::string &file_name);
    std::shared_ptr<Variable> find_variable(const char *name) const {
        for (int i = 0; i < variables_.size(); i++) {
            if (strcmp(name, variables_[i]->name().c_str()) == 0) {
                return variables_[i];
            }
        }
        return std::shared_ptr<Variable>();
    }
    std::shared_ptr<Surface> find_surface(int id) {
        for (int i = 0; i < surfaces_.size(); i++) {
            if (surfaces_[i]->get_id() == id)
                return surfaces_[i];
        }
        return std::shared_ptr<Surface>();
    }

private:
    void parse_thickness(const char *value, std::shared_ptr<Surface> surface_builder) const {
        if (value[0] == 0) {
            surface_builder->add_thickness(0.0);
            return;
        }
        if (isalpha(value[0])) {
            auto var = find_variable(value);
            if (var) {
                for (unsigned i = 0; i < var->num_scenarios(); i++) {
                    auto s = var->get_value(i);
                    auto d = strtod(s.c_str(), NULL);
                    surface_builder->add_thickness(d);
                }
            } else {
                fprintf(stderr, "Variable %s was not found\n", value);
                surface_builder->add_thickness(0.0);
            }
        } else {
            surface_builder->add_thickness(strtod(value, NULL));
        }
    }

public:
    const DescriptiveData &get_descriptive_data() const;
    const std::vector<std::shared_ptr<Variable>> &get_variables() const;
    const std::vector<std::shared_ptr<Surface>> &get_surfaces() const;
    const std::vector<std::shared_ptr<AsphericalData>> &get_aspherical_data() const;

private:
    DescriptiveData descriptive_data_;
    std::vector<std::shared_ptr<Variable>> variables_;
    std::vector<std::shared_ptr<Surface>> surfaces_;
    std::vector<std::shared_ptr<AsphericalData>> aspherical_data_;
};

// Sizeof buf must be == sizeof input
static bool parse_delimited(char *input_start, size_t input_size,
                            std::vector<const char *> &out_tokens, char *buf,
                            const char *delimiters) noexcept {
    out_tokens.clear();

    if (input_size == 0) {
        return true;
    }
    const char *input_ptr = input_start;
    const char *input_end = input_start + input_size;
    char *wordp = buf;

    while (*input_ptr && input_ptr != input_end) {
        char *word = wordp;
        *wordp = 0;

        bool inquote = false;
        while (*input_ptr && input_ptr != input_end) {
            if (word == wordp) {
                // we are at the beginning for a word, so look
                // for potential quote
                if (*input_ptr == '"' && !inquote) {
                    // We are in a quoted word
                    inquote = true;
                    input_ptr++;
                    continue;
                }
            }
            if (inquote) {
                // We are in a quoted word
                if (*input_ptr == '"') {
                    // Check if it is an escape - i.e.
                    // double quote
                    if (input_ptr + 1 < input_end && *(input_ptr + 1) == '"') {
                        // escape so we add the quote
                        // character
                        *wordp++ = '"';
                        input_ptr += 2;
                        continue;
                    } else {
                        // not escape so the quoted word
                        // ends here
                        inquote = false;
                        *wordp++ = 0;
                        input_ptr++;
                        if (input_ptr < input_end &&
                            (*input_ptr == ',' || *input_ptr == '\t' ||
                             (delimiters && strchr(delimiters, *input_ptr)))) {
                            // Skip delimiter
                            // following quote
                            input_ptr++;
                        }
                        break;
                    }
                } else {
                    // still in quoted word
                    *wordp++ = *input_ptr++;
                    continue;
                }
            } else {
                // Not in quoted word
                if (*input_ptr == ',' || *input_ptr == '\t' ||
                    (delimiters && strchr(delimiters, *input_ptr))) {
                    // word ends due to delimiter
                    *wordp++ = 0;
                    input_ptr++;
                    break;
                } else if (*input_ptr == '\r' || *input_ptr == '\n') {
                    // skip line feed or CRLF
                    *wordp++ = 0;
                    if (*input_ptr == '\r' && input_ptr + 1 < input_end &&
                        *(input_ptr + 1) == '\n') {
                        input_ptr++;
                    }
                    input_ptr++;
                    break;
                } else {
                    *wordp++ = *input_ptr++;
                }
            }
        }
        out_tokens.push_back(word);
    }
    return true;
}

enum Section {
    DESCRIPTIVE_DATA = 1,
    CONSTANTS = 2,
    VARIABLE_DISTANCES = 3,
    LENS_DATA = 4,
    ASPHERICAL_DATA = 5
};

struct SectionMapping {
    const char *name;
    int section;
};

static SectionMapping g_SectionMappings[] = {{"[descriptive data]", DESCRIPTIVE_DATA},
                                             {"[constants]", CONSTANTS},
                                             {"[variable distances]", VARIABLE_DISTANCES},
                                             {"[lens data]", LENS_DATA},
                                             {"[aspherical data]", ASPHERICAL_DATA},
                                             {NULL, 0}};

static int find_section(const char *name) {
    int section = 0;
    for (int i = 0; i < sizeof g_SectionMappings / sizeof(SectionMapping); i++) {
        if (g_SectionMappings[i].name == NULL) {
            section = g_SectionMappings[i].section;
            break;
        } else if (strcmp(g_SectionMappings[i].name, name) == 0) {
            section = g_SectionMappings[i].section;
            break;
        }
    }
    return section;
}

bool LensSystem::parse_file(const std::string &file_name) {

    FILE *fp = fopen(file_name.c_str(), "r");
    if (fp == NULL) {
        fprintf(stderr, "Unable to open file %s: %s\n", file_name.c_str(),
                strerror(errno));
        return false;
    }

    char line[256];                 // input line
    char buf[256];                  // for tokenizing
    std::vector<const char *> words;// tokenized words
    int current_section = 0;        // Current section

    while (fgets(line, sizeof line, fp) != NULL) {

        if (!parse_delimited(line, sizeof line, words, buf, "\t\n")) {
            continue;
        }
        if (words.size() == 0) {
            continue;
        }
        if (words[0][0] == '#') {
            // comment
            continue;
        }
        if (words[0][0] == '[') {
            // section name
            current_section = find_section(words[0]);
            continue;
        }

        switch (current_section) {
            case DESCRIPTIVE_DATA:
                if (words.size() >= 2 && strcmp(words[0], "title") == 0) {
                    descriptive_data_.set_title(words[1]);
                }
                break;
            case VARIABLE_DISTANCES:
                if (words.size() >= 2) {
                    auto var = std::make_shared<Variable>(words[0]);
                    for (int i = 1; i < words.size(); i++) {
                        var->add_value(words[i]);
                    }
                    variables_.push_back(var);
                }
                break;
            case LENS_DATA: {
                if (words.size() < 2)
                    break;
                int id = atoi(words[0]);
                auto surface_data = std::make_shared<Surface>(id);
                SurfaceType type = SurfaceType::surface;
                /* radius */
                if (strcmp(words[1], "AS") == 0) {
                    type = SurfaceType::aperture_stop;
                    surface_data->set_radius(0.0);
                } else if (strcmp(words[1], "CG") == 0) {
                    surface_data->set_radius(0.0);
                    surface_data->set_is_cover_glass(true);
                } else {
                    surface_data->set_radius(strtod(words[1], NULL));
                }
                surface_data->set_surface_type(type);
                /* thickness */

                if (words.size() >= 3 && strlen(words[2]) > 0) {
                    parse_thickness(words[2], surface_data);
                }
                /* refractive index */
                if (words.size() >= 4 && strlen(words[3]) > 0) {
                    surface_data->set_refractive_index(strtod(words[3], NULL));
                }
                /* diameter */
                if (words.size() >= 5 && strlen(words[4]) > 0) {
                    surface_data->set_diameter(strtod(words[4], NULL));
                }
                /* abbe vd */
                if (words.size() >= 6 && strlen(words[5]) > 0) {
                    surface_data->set_abbe_vd(strtod(words[5], NULL));
                }
                surfaces_.push_back(surface_data);
            } break;
            case ASPHERICAL_DATA: {
                int id = atoi(words[0]);
                auto aspherical_data = std::make_shared<AsphericalData>(id);
                for (int i = 1; i < words.size(); i++) {
                    aspherical_data->add_data(strtod(words[i], NULL));
                }
                aspherical_data_.push_back(aspherical_data);
                auto surface_builder = find_surface(id);
                if (!surface_builder) {
                    fprintf(stderr, "Ignoring aspherical data as no surface numbered %d\n",
                            id);
                } else {
                    surface_builder->set_aspherical_data(aspherical_data);
                }
            } break;
            default:
                break;
        }
    }

    fclose(fp);
    return true;
}
const DescriptiveData &LensSystem::get_descriptive_data() const {
    return descriptive_data_;
}
const std::vector<std::shared_ptr<Variable>> &LensSystem::get_variables() const {
    return variables_;
}
const std::vector<std::shared_ptr<Surface>> &LensSystem::get_surfaces() const {
    return surfaces_;
}
const std::vector<std::shared_ptr<AsphericalData>> &LensSystem::get_aspherical_data() const {
    return aspherical_data_;
}

class RayOptGenerator {
public:
    void generate_preamble(FILE *fp) {
        fputs(
                "%pylab inline\n"
                "import warnings\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import rayopt as ro\n"
                "warnings.simplefilter(\"ignore\", FutureWarning)\n"
                "np.seterr(divide=\"ignore\", invalid=\"ignore\")\n"
                "np.set_printoptions(precision=4)\n",
                fp);
    }
    void generate_description(const LensSystem &system, FILE *fp) {
        auto descriptive_data = system.get_descriptive_data();
        auto title = descriptive_data.get_title();
        fprintf(fp, "description = \"%s\"\n", title.c_str());
    }

    double get_thickness(const LensSystem &system, unsigned id, unsigned scenario) {
        if (id == 0) {
            return 20.0;
        }
        auto surfaces = system.get_surfaces();
        // TODO d0
        auto s = surfaces[id - 1];
        return s->get_thickness(scenario);
    }

    void generate_lens_data(const LensSystem &system, unsigned scenario, FILE *fp) {
        auto surfaces = system.get_surfaces();
        auto view_angles = system.find_variable("Angle of View");
        auto image_heights = system.find_variable("Image Height");
        auto back_focus = system.find_variable("Bf");

        if (scenario >= view_angles->num_scenarios() ||
            scenario >= image_heights->num_scenarios() ||
            scenario >= back_focus->num_scenarios()) {
            fprintf(stderr, "Scenario %u has missing data\n", scenario);
            return;
        }

        fputs("columns = \"type distance roc diameter material\"\n", fp);
        fprintf(fp, "# number of surfaces = %u\n", (unsigned) surfaces.size());
        fputs("lensdata = \"\"\"\n", fp);
        fprintf(fp, "O 0.0 0.0 %f AIR\n", image_heights->get_value_as_double(scenario) * 1.3);
        double Bf = back_focus->get_value_as_double(scenario);
        for (unsigned i = 0; i < surfaces.size(); i++) {
            auto s = surfaces[i];
            if (i + 1 == surfaces.size() && s->is_cover_glass()) {
                // Oddity - override the Bf
                Bf = s->get_thickness(scenario);
            }
            const char *type = s->get_surface_type() == SurfaceType::surface ? "S" : "A";
            if (s->get_refractive_index() != 0.0) {
                fprintf(fp, "%s %f %f %f %f/%f\n",
                        type,
                        get_thickness(system, i, scenario),
                        s->get_radius(),
                        s->get_diameter(),
                        s->get_refractive_index(),
                        s->get_abbe_vd());
            } else {
                fprintf(fp, "%s %f %f %f AIR\n",
                        type,
                        get_thickness(system, i, scenario),
                        s->get_radius(),
                        s->get_diameter());
            }
        }
        fprintf(fp, "I %f 0 %f AIR\n", Bf, image_heights->get_value_as_double(scenario));
        fputs("\"\"\"\n", fp);
    }

    void generate_system(const LensSystem &system, unsigned scenario, FILE *fp) {
        auto view_angles = system.find_variable("Angle of View");
        fputs("s = ro.system_from_text(text, columns.split(),\n"
                "    description=description)\n",
                fp);
        fputs("s.fields = 0, .7, 1.", fp);
        fprintf(fp, "s.object.angle = np.deg2rad(%f)\n", view_angles->get_value_as_double(scenario) / 2.0);
    }

    void generate_aspherics(const LensSystem &system, FILE *fp) {
        auto aspheres = system.get_aspherical_data();

    }

    void generate(const LensSystem &system, unsigned scenario = 0, FILE *fp = stdout) {
        generate_preamble(fp);
        generate_description(system, fp);
        generate_lens_data(system, scenario, fp);
        generate_system(system, scenario, fp);
    }
};

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Please provide a file name");
        exit(1);
    }
    LensSystem system;
    system.parse_file(argv[1]);
    RayOptGenerator generator;
    generator.generate(system);
    return 0;
}
