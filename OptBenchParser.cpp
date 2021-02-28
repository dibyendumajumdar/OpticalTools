#include <memory>
#include <string>
#include <vector>

#include <cassert>
#include <cmath>
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
    // K is 1, A_4 is 2, ...
    double data(int i) const {
        return i >= 0 && i < data_.size() ? data_[i] : 0.0;
    }
    int get_surface_number() const {
        return surface_number_;
    }
    void dump(FILE *fp) {
        fprintf(fp, "Aspheric values[%d] = ", surface_number_);
        for (int i = 0; i < data_points(); i++) {
            fprintf(fp, "%.12g ", data(i));
        }
        fputc('\n', fp);
    }

private:
    int surface_number_;
    std::vector<double> data_;
};

enum SurfaceType { surface,
                   aperture_stop,
                   field_stop
};
const char *SurfaceTypeNames[] = {
        "S", "AS", "FS"};

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
    double get_thickness(unsigned scenario) const {
        if (scenario < thickness_by_scenario_.size())
            return thickness_by_scenario_[scenario];
        else {
            assert(1 == thickness_by_scenario_.size());
            return thickness_by_scenario_[0];
        }
    }
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
    void dump(FILE *fp, unsigned scenario = 0) {
        fprintf(fp, "Surface[%d] = type=%s radius=%.12g thickness=%.12g diameter = %.12g nd = %.12g vd = %.12g\n",
                id_, SurfaceTypeNames[surface_type_], radius_, get_thickness(scenario), diameter_, refractive_index_, abbe_vd_);
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
    void dump(FILE *fp = stdout, unsigned scenario = 0) {
        for (int i = 0; i < surfaces_.size(); i++) {
            surfaces_.at(i)->dump(fp, scenario);
            if (surfaces_.at(i)->get_aspherical_data()) {
                surfaces_.at(i)->get_aspherical_data()->dump(fp);
            }
        }
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
                            (// *input_ptr == ',' || *input_ptr == '\t' ||
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
                if (//*input_ptr == ',' || *input_ptr == '\t' ||
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
    int surface_id = 1;             // We used to read the id from the OptBench data but this doesn't always work

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
                int id = surface_id++;
                auto surface_data = std::make_shared<Surface>(id);
                SurfaceType type = SurfaceType::surface;
                /* radius */
                if (strcmp(words[1], "AS") == 0) {
                    type = SurfaceType::aperture_stop;
                    surface_data->set_radius(0.0);
                } else if (strcmp(words[1], "FS") == 0) {
                    type = SurfaceType::field_stop;
                    surface_data->set_radius(0.0);
                } else if (strcmp(words[1], "CG") == 0) {
                    surface_data->set_radius(0.0);
                    surface_data->set_is_cover_glass(true);
                } else {
                    if (strcmp(words[1], "Infinity") == 0)
                        surface_data->set_radius(0.0);
                    else
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
        // rayopt requires that we put the thickness of element at post 1
        // on element at pos+1. So that means that when we are requested
        // pos, we need to get pos-1.
        // Now pos - 1 could be an FS, in which case we need to back up
        // and get pos - 2 and and the FS thickness to pos - 2 because we can't
        // deal with FS.
        auto s = surfaces[id - 1];
        double fs = 0.0;
        if (s->get_surface_type() == SurfaceType::field_stop) {
            //s->dump(stderr);
            if (s->get_id() < 1) {
                fprintf(stderr, "Bad data at surface %d, \n", s->get_id());
                exit(1);
            }
            // Add the field stop to the thickness
            fs = s->get_thickness(scenario);
            s = surfaces[id - 2];
            //s->dump(stderr);
            //fprintf(stderr, "Will add FS thickness %.12g to element\n",  fs);
        }
        double thickness = fs + s->get_thickness(scenario);
        // print thickness
        return thickness;
    }

    /* handling of Field Stop surface is problematic because it messes up the
     * numbering of surfaces and therefore we need to adjust the surface id
     * when we see a field stop. Currently we cannot handle more than 1 field stop.
     */
    void generate_lens_data(const LensSystem &system, unsigned scenario, unsigned *fs, FILE *fp) {
        auto surfaces = system.get_surfaces();
        auto view_angles = system.find_variable("Angle of View");
        auto image_heights = system.find_variable("Image Height");
        auto back_focus = system.find_variable("Bf");
        auto aperture_diameters = system.find_variable("Aperture Diameter");

        if (scenario >= view_angles->num_scenarios() ||
            scenario >= image_heights->num_scenarios() ||
            scenario >= back_focus->num_scenarios() ||
            (aperture_diameters && scenario >= aperture_diameters->num_scenarios())) {
            fprintf(stderr, "Scenario %u has missing data\n", scenario);
            return;
        }

        fputs("columns = \"type distance roc diameter material\"\n", fp);
        fprintf(fp, "# number of surfaces = %u\n", (unsigned) surfaces.size());
        fputs("lensdata = \"\"\"\n", fp);
        fprintf(fp, "O 0.0 0.0 %.12g AIR\n", surfaces[0]->get_diameter() * 1.3);
        double Bf = back_focus->get_value_as_double(scenario);
        *fs = 0;
        for (unsigned i = 0; i < surfaces.size(); i++) {
            auto s = surfaces[i];
            if (s->get_surface_type() == SurfaceType::field_stop) {
                if (*fs == 0)
                    *fs = (unsigned) s->get_id();
                else {
                    //fprintf(stderr, "Cannot process second surface of type FS\n");
                    //s->dump(stderr);
                }
                continue;
            }
            if (i + 1 == surfaces.size() && s->is_cover_glass()) {
                // Oddity - override the Bf
                Bf = s->get_thickness(scenario);
            }
            //s->dump(stderr, scenario);
            const char *type = s->get_surface_type() == SurfaceType::surface ? "S" : "A";
            double diameter = s->get_diameter();
            if (s->get_surface_type() == SurfaceType::aperture_stop && aperture_diameters) {
                diameter = aperture_diameters->get_value_as_double(scenario);
            }
            if (s->get_refractive_index() != 0.0) {
                fprintf(fp, "%s %.12g %.12g %.12g %.12g/%.12g\n",
                        type,
                        get_thickness(system, i, scenario),
                        s->get_radius(),
                        diameter,
                        s->get_refractive_index(),
                        s->get_abbe_vd());
            } else {
                fprintf(fp, "%s %.12g %.12g %.12g AIR\n",
                        type,
                        get_thickness(system, i, scenario),
                        s->get_radius(),
                        diameter);
            }
        }
        fprintf(fp, "I %.12g 0 %.12g AIR\n", Bf, image_heights->get_value_as_double(scenario));
        fputs("\"\"\"\n", fp);
    }

    void generate_system(const LensSystem &system, unsigned scenario, FILE *fp) {
        auto view_angles = system.find_variable("Angle of View");
        fputs("s = ro.system_from_text(lensdata, columns.split(),\n"
              "    description=description)\n",
              fp);
        fputs("s.fields = 0, .7, 1.\n", fp);
        fprintf(fp, "s.object.angle = np.deg2rad(%f)\n", view_angles->get_value_as_double(scenario) / 2.0);
    }

    void generate_aspherics(const LensSystem &system, unsigned fs, FILE *fp) {
        auto aspheres = system.get_aspherical_data();
        for (unsigned i = 0; i < aspheres.size(); i++) {
            auto asphere = aspheres[i];
            /* If there was a field stop then our aspheric indices will be out by 1 */
            int id = fs != 0 && asphere->get_surface_number() > fs ? asphere->get_surface_number() - 1 : asphere->get_surface_number();// Adjust for skipped field stop
            fprintf(fp, "s[%d].conic = %.12g\n", id, asphere->data(1));
            fprintf(fp, "s[%d].aspherics = [0, %.12g, %.12g, %.12g, %.12g, %.12g, %.12g]\n",
                    id,
                    asphere->data(2), asphere->data(3), asphere->data(4),
                    asphere->data(5), asphere->data(6), asphere->data(7));
        }
    }

    void generate_rest(const LensSystem &system, FILE *fp) {
        fputs("s.update()\n"
              "print(s)\n"
              "ro.Analysis(s)\n",
              fp);
    }

    void generate(const LensSystem &system, unsigned scenario = 0, FILE *fp = stdout) {
        unsigned fs = 0;
        if (scenario == 0)
            generate_preamble(fp);
        generate_description(system, fp);
        generate_lens_data(system, scenario, &fs, fp);
        generate_system(system, scenario, fp);
        generate_aspherics(system, fs, fp);
        generate_rest(system, fp);
    }
};

class KDPGenerator {
public:
    double get_thickness(const LensSystem &system, unsigned id, unsigned scenario) {
        auto surfaces = system.get_surfaces();
        auto s = surfaces[id];
        double fs = 0.0;
        if (s->get_surface_type() == SurfaceType::field_stop) {
            //s->dump(stderr);
            if (s->get_id() == 0) {
                fprintf(stderr, "Bad data at surface %d, \n", s->get_id());
                exit(1);
            }
            // Add the field stop to the thickness
            fs = s->get_thickness(scenario);
            s = surfaces[id - 1];
            //s->dump(stderr);
            //fprintf(stderr, "Will add FS thickness %.12g to element\n",  fs);
        }
        double thickness = fs + s->get_thickness(scenario);
        // print thickness
        return thickness;
    }
    void generate_preamble(const LensSystem &system, FILE *fp) {
        auto descriptive_data = system.get_descriptive_data();
        auto title = descriptive_data.get_title();
        fprintf(fp,
                "LENS\n"
                "LI,%s\n"
                "WV,0.58756,0.48613,0.65627,0.0,0.0\n"
                "UNITS MM\n",
                title.c_str());
    }
    void generate_system(const LensSystem &system, unsigned scenario, FILE *fp) {
        auto view_angles = system.find_variable("Angle of View");
        fprintf(fp, "SCY FANG %.12g\n", view_angles->get_value_as_double(scenario) / 2.0);
    }
    void generate_object_conjugate(const LensSystem &system, unsigned scenario, FILE *fp) {
        fprintf(fp, "C Object conjugate TODO\n");
        fputs("TH 1.0E20\n"
              "AIR\n"
              "AIR\n",
              fp);
    }
    void generate_aspherics(const std::shared_ptr<AsphericalData> asphere, FILE *fp) {
        fprintf(fp, "CC %.12g\n", asphere->data(1));
        fprintf(fp, "ASPH,%.12g,%.12g,%.12g,%.12g,0.0\n",
                asphere->data(2), asphere->data(3), asphere->data(4),
                asphere->data(5), asphere->data(6), asphere->data(7));
        if (asphere->data_points() > 6) {
            fprintf(fp, "ASPH2,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                    asphere->data(6), asphere->data(7), asphere->data(8),
                    asphere->data(9), asphere->data(10));
        }
    }

    /* handling of Field Stop surface is problematic because it messes up the
    * numbering of surfaces and therefore we need to adjust the surface id
    * when we see a field stop. Currently we cannot handle more than 1 field stop.
    */
    void generate_lens_data(const LensSystem &system, unsigned scenario, unsigned *fs, FILE *fp) {
        auto surfaces = system.get_surfaces();
        auto view_angles = system.find_variable("Angle of View");
        auto image_heights = system.find_variable("Image Height");
        auto back_focus = system.find_variable("Bf");
        auto aperture_diameters = system.find_variable("Aperture Diameter");

        if (scenario >= view_angles->num_scenarios() ||
            scenario >= image_heights->num_scenarios() ||
            scenario >= back_focus->num_scenarios() ||
            (aperture_diameters && scenario >= aperture_diameters->num_scenarios())) {
            fprintf(stderr, "Scenario %u has missing data\n", scenario);
            return;
        }
        fprintf(fp, "C number of surfaces = %u\n", (unsigned) surfaces.size());
        double Bf = back_focus->get_value_as_double(scenario);
        *fs = 0;
        for (unsigned i = 0; i < surfaces.size(); i++) {
            auto s = surfaces[i];
            if (s->get_surface_type() == SurfaceType::field_stop) {
                if (*fs == 0)
                    *fs = (unsigned) s->get_id();
                else {
                    fprintf(stderr, "Cannot process second surface of type FS\n");
                    s->dump(stderr);
                }
                continue;
            }
            if (i + 1 == surfaces.size() && s->is_cover_glass()) {
                // Oddity - override the Bf
                Bf = s->get_thickness(scenario);
            }
            const char *type = s->get_surface_type() == SurfaceType::surface ? "S" : "A";
            double diameter = s->get_diameter();
            if (s->get_surface_type() == SurfaceType::aperture_stop && aperture_diameters) {
                diameter = aperture_diameters->get_value_as_double(scenario);
            }
            fprintf(fp, "C THE FOLLOWING DATA REFERS TO SURFACE #%d\n", s->get_id());
            if (s->get_surface_type() == SurfaceType::surface) {
                fprintf(fp, "RD %.12g\n", s->get_radius());
                fprintf(fp, "TH %.12g\n", get_thickness(system, i, scenario));
                fprintf(fp, "CLAP %.12g\n", diameter / 2.0);
                auto aspherics = s->get_aspherical_data();
                if (aspherics) {
                    generate_aspherics(aspherics, fp);
                }
                if (s->get_refractive_index() != 0.0) {
                    fprintf(fp, "MODEL G%d,%.12g,%.12g\n", s->get_id(), s->get_refractive_index(), s->get_abbe_vd());
                } else {
                    fprintf(fp, "AIR\n");
                }
            } else if (s->get_surface_type() == SurfaceType::aperture_stop) {
                fprintf(fp, "TH %.12g\n", get_thickness(system, i, scenario));
                fprintf(fp, "CLAP %.12g\n", diameter / 2.0);
                fprintf(fp, "REFS\nASTOP\nAIR\n");
            }
        }
    }
    void generate_rest(FILE *fp) {
        fputs("AIR\nAIR\n"
              "EOS\n"
              "LEPRT\n",
              fp);
    }
    void generate(const LensSystem &system, unsigned scenario = 0, FILE *fp = stdout) {
        unsigned fs = 0;
        generate_preamble(system, fp);
        generate_system(system, scenario, fp);
        generate_object_conjugate(system, scenario, fp);
        generate_lens_data(system, scenario, &fs, fp);
        generate_rest(fp);
    }
};

class RayGenerator {
public:
    double get_thickness(const LensSystem &system, unsigned id, unsigned scenario) {
        auto surfaces = system.get_surfaces();
        auto s = surfaces[id];
        double fs = 0.0;
        if (s->get_surface_type() == SurfaceType::field_stop) {
            //s->dump(stderr);
            if (s->get_id() == 0) {
                fprintf(stderr, "Bad data at surface %d, \n", s->get_id());
                exit(1);
            }
            // Add the field stop to the thickness
            fs = s->get_thickness(scenario);
            s = surfaces[id - 1];
            //s->dump(stderr);
            //fprintf(stderr, "Will add FS thickness %.12g to element\n",  fs);
        }
        double thickness = fs + s->get_thickness(scenario);
        // print thickness
        return thickness;
    }
    void generate_preamble(const LensSystem &system, FILE *fp) {
        auto descriptive_data = system.get_descriptive_data();
        auto title = descriptive_data.get_title();
        fprintf(fp,
                "System %s\nDigits 5 0.00000001\n",
                title.c_str());
    }
    double getRefractiveIndexRatio(const LensSystem &system, unsigned i) {
        auto surfaces = system.get_surfaces();
        if (i == 0) {
            return 1.0 / surfaces[i]->get_refractive_index();
        } else {
            unsigned prev = i - 1;
            if (surfaces[prev]->get_surface_type() == SurfaceType::aperture_stop) {
                prev--;
            }
            // 1.0 is AIR
            double N = surfaces[prev]->get_refractive_index() != 0.0 ? surfaces[prev]->get_refractive_index() : 1.0;
            double N1 = surfaces[i]->get_refractive_index() != 0.0 ? surfaces[i]->get_refractive_index() : 1.0;
            double ratio = N/N1 ;
            return ratio;
        }
    }
    /* handling of Field Stop surface is problematic because it messes up the
    * numbering of surfaces and therefore we need to adjust the surface id
    * when we see a field stop. Currently we cannot handle more than 1 field stop.
    */
    void generate_lens_data(const LensSystem &system, unsigned scenario, unsigned *fs, FILE *fp) {
        auto surfaces = system.get_surfaces();
        auto view_angles = system.find_variable("Angle of View");
        auto image_heights = system.find_variable("Image Height");
        auto back_focus = system.find_variable("Bf");
        auto aperture_diameters = system.find_variable("Aperture Diameter");

        if (scenario >= view_angles->num_scenarios() ||
            scenario >= image_heights->num_scenarios() ||
            scenario >= back_focus->num_scenarios() ||
            (aperture_diameters && scenario >= aperture_diameters->num_scenarios())) {
            fprintf(stderr, "Scenario %u has missing data\n", scenario);
            return;
        }
        double Bf = back_focus->get_value_as_double(scenario);
        *fs = 0;
        double thickness = 0;
        for (unsigned i = 0; i < surfaces.size(); i++) {
            auto s = surfaces[i];
            if (s->get_surface_type() == SurfaceType::field_stop) {
                if (*fs == 0)
                    *fs = (unsigned) s->get_id();
                else {
                    fprintf(stderr, "Cannot process second surface of type FS\n");
                    s->dump(stderr);
                }
                continue;
            }
            if (i + 1 == surfaces.size() && s->is_cover_glass()) {
                // Oddity - override the Bf
                Bf = s->get_thickness(scenario);
            }
            const char *type = s->get_surface_type() == SurfaceType::surface ? "S" : "A";
            double diameter = s->get_diameter();
            if (s->get_surface_type() == SurfaceType::aperture_stop && aperture_diameters) {
                diameter = aperture_diameters->get_value_as_double(scenario);
            }
            //fprintf(fp, "# THE FOLLOWING DATA REFERS TO SURFACE #%d\n", s->get_id());
            double c = s->get_radius() != 0.0 ? 1.0/s->get_radius(): 0.0;
            double mu = s->get_surface_type() == SurfaceType::surface ? getRefractiveIndexRatio(system, i) : 1.0; // For aperture stop set mu to 1.0
            double x = thickness;
            thickness += get_thickness(system, i, scenario);
            auto aspherics = s->get_aspherical_data();
            if (aspherics) {
                double k = aspherics->data(1)+1.0;
                fprintf(stdout, "rayAddSurface S%d %g %g %g %g %g %g 0 0 0 0 0 %g %g %g %g %g\n", i, c, k, 0.0, aspherics->data(2),
                        mu, x, aspherics->data(3), aspherics->data(4), aspherics->data(5), aspherics->data(6), aspherics->data(7));
            }
            else {
                fprintf(stdout, "rayAddSurface S%d %g %g %g %g %g %g 0 0 0 0 0 %g %g %g %g %g\n", i, c, 1.0, 0.0, 0.0,
                        mu, x, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
        }
    }
    void generate_rest(FILE *fp) {
        fputs("rayPrtSystem\n"
              "rayGenerator \"spherical\"@[-150000, 0, 0] & [1, 0, 0]0.00020 50000, 2, 4, 4	0.1, -0.001	2, 13 \"bundle\"\n"
              "rayTrace\n"
              "rayGetFoci\n"
              "rayPrtFoci\n"
              "Quit\n",
              fp);
    }
    void generate(const LensSystem &system, unsigned scenario = 0, FILE *fp = stdout) {
        unsigned fs = 0;
        generate_preamble(system, fp);
        generate_lens_data(system, scenario, &fs, fp);
        generate_rest(fp);
    }
};

class RayOpticsGenerator {
public:
    double get_thickness(const LensSystem &system, unsigned id, unsigned scenario) {
        auto surfaces = system.get_surfaces();
        auto s = surfaces[id];
        double fs = 0.0;
        if (s->get_surface_type() == SurfaceType::field_stop) {
            //s->dump(stderr);
            if (s->get_id() == 0) {
                fprintf(stderr, "Bad data at surface %d, \n", s->get_id());
                exit(1);
            }
            // Add the field stop to the thickness
            fs = s->get_thickness(scenario);
            s = surfaces[id - 1];
            //s->dump(stderr);
            //fprintf(stderr, "Will add FS thickness %.12g to element\n",  fs);
        }
        double thickness = fs + s->get_thickness(scenario);
        // print thickness
        return thickness;
    }

    double get_angle_of_view(const LensSystem &system, unsigned scenario) {
        auto view_angles = system.find_variable("Angle of View");
        return view_angles->get_value_as_double(scenario) / 2.0;
    }
    void generate_preamble(const LensSystem &system, unsigned scenario, FILE *fp) {
        auto descriptive_data = system.get_descriptive_data();
        auto title = descriptive_data.get_title();
        auto f_number = system.find_variable("F-Number");
        fprintf(fp,
                "%%matplotlib inline\n"
                "isdark = False\n"
                "from rayoptics.environment import *\n"
                "from rayoptics.elem.elements import Element\n"
                "from rayoptics.raytr.trace import apply_paraxial_vignetting\n"
                "\n"
                "# %s\n"
                "# Obtained via https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm\n"
                "\n"
                "opm = OpticalModel()\n"
                "sm  = opm.seq_model\n"
                "osp = opm.optical_spec\n"
                "pm = opm.parax_model\n"
                "osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=%g)\n"
                "osp.field_of_view = FieldSpec(osp, key=['object', 'angle'], flds=[0., %g])\n"
                "osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)\n"
                "opm.system_spec.title = '%s'\n"
                "opm.system_spec.dimensions = 'MM'\n"
                "opm.radius_mode = True\n",
                title.c_str(),
                f_number->get_value_as_double(scenario),
                get_angle_of_view(system, scenario),
                title.c_str());
    }
    void generate_aspherics(const std::shared_ptr<AsphericalData> asphere, FILE *fp) {
        fprintf(fp, "sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=%g, cc=%g,\n", asphere->data(0), asphere->data(1));
        fprintf(fp, "\tcoefs=[0.0,%g,%g,%g,%g,%g,%g])\n",
                asphere->data(2), asphere->data(3), asphere->data(4),
                asphere->data(5), asphere->data(6), asphere->data(7));
    }
    /* handling of Field Stop surface is problematic because it messes up the
    * numbering of surfaces and therefore we need to adjust the surface id
    * when we see a field stop. Currently we cannot handle more than 1 field stop.
    */
    void generate_lens_data(const LensSystem &system, unsigned scenario, unsigned *fs, FILE *fp) {
        auto surfaces = system.get_surfaces();
        auto view_angles = system.find_variable("Angle of View");
        auto image_heights = system.find_variable("Image Height");
        auto back_focus = system.find_variable("Bf");
        auto aperture_diameters = system.find_variable("Aperture Diameter");

        if (scenario >= view_angles->num_scenarios() ||
            scenario >= image_heights->num_scenarios() ||
            scenario >= back_focus->num_scenarios() ||
            (aperture_diameters && scenario >= aperture_diameters->num_scenarios())) {
            fprintf(stderr, "Scenario %u has missing data\n", scenario);
            return;
        }
        fprintf(fp, "sm.gaps[0].thi=1e10\n");
        double Bf = back_focus->get_value_as_double(scenario);
        *fs = 0;
        for (unsigned i = 0; i < surfaces.size(); i++) {
            auto s = surfaces[i];
            if (s->get_surface_type() == SurfaceType::field_stop) {
                if (*fs == 0)
                    *fs = (unsigned) s->get_id();
                else {
                    fprintf(stderr, "Cannot process second surface of type FS\n");
                    s->dump(stderr);
                }
                continue;
            }
            if (i + 1 == surfaces.size() && s->is_cover_glass()) {
                // Oddity - override the Bf
                Bf = s->get_thickness(scenario);
            }
            const char *type = s->get_surface_type() == SurfaceType::surface ? "S" : "A";
            double diameter = s->get_diameter();
            if (s->get_surface_type() == SurfaceType::aperture_stop && aperture_diameters) {
                diameter = aperture_diameters->get_value_as_double(scenario);
            }
            if (s->get_surface_type() == SurfaceType::surface) {
                if (s->get_refractive_index() != 0.0) {
                    fprintf(fp, "sm.add_surface([%g,%g,%g,%g])\n", s->get_radius(),
                            get_thickness(system, i, scenario),
                            s->get_refractive_index(),
                            s->get_abbe_vd());
                }
                else {
                    fprintf(fp, "sm.add_surface([%g,%g])\n", s->get_radius(),
                            get_thickness(system, i, scenario));
                }
                auto aspherics = s->get_aspherical_data();
                if (aspherics) {
                    generate_aspherics(aspherics, fp);
                }
            } else if (s->get_surface_type() == SurfaceType::aperture_stop) {
                fprintf(fp, "sm.add_surface([%g,%g])\n", s->get_radius(),
                        get_thickness(system, i, scenario));
                fprintf(fp, "sm.set_stop()\n");
            }
            fprintf(fp, "sm.ifcs[sm.cur_surface].max_aperture = %g\n", diameter / 2.0);
        }
    }
    void generate_rest(FILE *fp) {
        fputs("sm.list_surfaces()\n"
              "sm.list_gaps()\n"
              "opm.update_model()\n"
              "apply_paraxial_vignetting(opm)\n"
              "layout_plt = plt.figure(FigureClass=InteractiveLayout, opt_model=opm, do_draw_rays=True, do_paraxial_layout=False,\n"
              "                        is_dark=isdark).plot()\n"
              "sm.list_model()\n"
              "# List the optical specifications\n"
              "pm.first_order_data()\n"
              "# List the paraxial model\n"
              "pm.list_lens()\n"
              "# Plot the transverse ray aberrations\n"
              "abr_plt = plt.figure(FigureClass=RayFanFigure, opt_model=opm,\n"
              "          data_type='Ray', scale_type=Fit.All_Same, is_dark=isdark).plot()\n"
              "# Plot the wavefront aberration\n"
              "wav_plt = plt.figure(FigureClass=RayFanFigure, opt_model=opm,\n"
              "          data_type='OPD', scale_type=Fit.All_Same, is_dark=isdark).plot()\n"
              "# Plot spot diagrams\n"
              "spot_plt = plt.figure(FigureClass=SpotDiagramFigure, opt_model=opm, \n"
              "                      scale_type=Fit.User_Scale, user_scale_value=0.1, is_dark=isdark).plot()\n",
              fp);
    }
    void generate(const LensSystem &system, unsigned scenario = 0, FILE *fp = stdout) {
        unsigned fs = 0;
        generate_preamble(system, scenario, fp);
        generate_lens_data(system, scenario, &fs, fp);
        generate_rest(fp);
    }
};


int main(int argc, const char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Please provide a file name");
        exit(1);
    }
    unsigned scenario = 0;
    if (argc == 3) {
        scenario = atoi(argv[2]);
    }
    LensSystem system;
    system.parse_file(argv[1]);
    system.dump(stdout, scenario);
    //RayOptGenerator generator;
    //KDPGenerator generator;
//    RayGenerator generator;

    RayOpticsGenerator generator;
    generator.generate(system, scenario);
    return 0;
}
