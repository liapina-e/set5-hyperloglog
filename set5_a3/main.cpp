#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <numeric>

class RandomStreamGen {
private:
    std::mt19937_64 rng;
    std::uniform_int_distribution<int> char_dist;
    std::uniform_int_distribution<int> length_dist;
    std::bernoulli_distribution duplicate_dist;

    const std::string alphabet =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-";

    double duplicate_probability;
    int max_unique_strings;

public:
    RandomStreamGen(double dup_prob = 0.3, int max_unique = 1000000, uint64_t seed = 0) : duplicate_probability(dup_prob), max_unique_strings(max_unique) {

        if (seed == 0) {
            seed = std::chrono::steady_clock::now().time_since_epoch().count();
        }
        rng.seed(seed);

        char_dist = std::uniform_int_distribution<int>(0, alphabet.size() - 1);
        length_dist = std::uniform_int_distribution<int>(5, 30);
        duplicate_dist = std::bernoulli_distribution(duplicate_probability);
    }

    std::string generate_string() {
        int length = length_dist(rng);
        std::string result;
        result.reserve(length);

        for (int i = 0; i < length; ++i) {
            result.push_back(alphabet[char_dist(rng)]);
        }

        return result;
    }

    std::vector<std::string> generate_stream(int total_size) {
        std::vector<std::string> stream;
        stream.reserve(total_size);

        std::unordered_set<std::string> unique_strings_in_this_stream;

        for (int i = 0; i < total_size; ++i) {
            if (duplicate_dist(rng) && !unique_strings_in_this_stream.empty()) {
                auto it = unique_strings_in_this_stream.begin();
                std::advance(it, std::uniform_int_distribution<int>(
                    0, unique_strings_in_this_stream.size() - 1)(rng));
                stream.push_back(*it);
            } else {
                std::string new_str;
                int attempts = 0;
                do {
                    new_str = generate_string();
                    attempts++;
                } while (unique_strings_in_this_stream.count(new_str) > 0 && attempts < 10);

                unique_strings_in_this_stream.insert(new_str);
                stream.push_back(new_str);
            }
        }

        return stream;
    }

    RandomStreamGen create_independent_generator(uint64_t seed_addition = 0) {
        uint64_t new_seed = std::chrono::steady_clock::now().time_since_epoch().count() + seed_addition;
        return RandomStreamGen(duplicate_probability, max_unique_strings, new_seed);
    }

    std::vector<std::vector<std::string>> split_by_time(
        const std::vector<std::string>& stream,
        const std::vector<int>& percentages) {

        std::vector<std::vector<std::string>> result;
        result.reserve(percentages.size());

        int total_size = stream.size();
        int current_idx = 0;

        for (int percentage : percentages) {
            int chunk_size = (total_size * percentage) / 100;
            if (current_idx + chunk_size > total_size) {
                chunk_size = total_size - current_idx;
            }

            std::vector<std::string> chunk(
                stream.begin() + current_idx,
                stream.begin() + current_idx + chunk_size
            );

            result.push_back(std::move(chunk));
            current_idx += chunk_size;

            if (current_idx >= total_size) break;
        }

        if (current_idx < total_size) {
            std::vector<std::string> remainder(stream.begin() + current_idx, stream.end());
            result.push_back(std::move(remainder));
        }

        return result;
    }

    std::vector<std::vector<std::string>> split_equal_parts(
        const std::vector<std::string>& stream,
        int part_percentage) {

        std::vector<int> percentages;
        int remaining = 100;

        while (remaining > 0) {
            int chunk = std::min(part_percentage, remaining);
            percentages.push_back(chunk);
            remaining -= chunk;
        }

        return split_by_time(stream, percentages);
    }
};

class HashFuncGen {
public:
    enum HashType {
        POLYNOMIAL,
        FNV1A,
        MURMUR
    };

private:
    uint64_t poly_p;
    uint64_t poly_mod;
    std::vector<uint64_t> p_powers;

    HashType hash_type;

    static constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    static constexpr uint64_t MURMUR_M = 0xc6a4a7935bd1e995ULL;
    static constexpr uint64_t MURMUR_R = 47;

public:
    HashFuncGen(HashType type = POLYNOMIAL) : hash_type(type) {
        if (type == POLYNOMIAL) {
            poly_p = 31;
            poly_mod = 1ULL << 32;
            p_powers.resize(31);
            p_powers[0] = 1;
            for (int i = 1; i <= 30; ++i) {
                p_powers[i] = (p_powers[i-1] * poly_p) % poly_mod;
            }
        }
    }

    uint32_t murmur_hash(const std::string& str) const {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(str.data());
        size_t len = str.size();
        uint64_t h = 0 ^ (len * MURMUR_M);

        while (len >= 8) {
            uint64_t k = *reinterpret_cast<const uint64_t*>(data);
            k *= MURMUR_M;
            k ^= k >> MURMUR_R;
            k *= MURMUR_M;
            h ^= k;
            h *= MURMUR_M;
            data += 8;
            len -= 8;
        }

        switch (len) {
            case 7: h ^= static_cast<uint64_t>(data[6]) << 48;
            case 6: h ^= static_cast<uint64_t>(data[5]) << 40;
            case 5: h ^= static_cast<uint64_t>(data[4]) << 32;
            case 4: h ^= static_cast<uint64_t>(data[3]) << 24;
            case 3: h ^= static_cast<uint64_t>(data[2]) << 16;
            case 2: h ^= static_cast<uint64_t>(data[1]) << 8;
            case 1: h ^= static_cast<uint64_t>(data[0]);
                h *= MURMUR_M;
        };

        h ^= h >> MURMUR_R;
        h *= MURMUR_M;
        h ^= h >> MURMUR_R;

        return static_cast<uint32_t>(h);
    }

    uint32_t operator()(const std::string& str) const {
        return murmur_hash(str);
    }
};

class HyperLogLog {
private:
    int b;
    int m;
    std::vector<uint8_t> registers;
    HashFuncGen hash_func;

    double alpha_m;

    double get_alpha_m() {
        if (m == 16) return 0.673;
        if (m == 32) return 0.697;
        if (m == 64) return 0.709;
        return 0.7213 / (1.0 + 1.079 / m);
    }

    uint32_t extract_index(uint32_t hash, int b) {
        return hash >> (32 - b);
    }

    uint8_t count_leading_zeros(uint32_t hash, int b) {
        uint32_t mask = (1 << (32 - b)) - 1;
        uint32_t value = hash & mask;

        if (value == 0) return 32 - b + 1;

        uint8_t count = 1;
        while ((value & (1 << (31 - b - count))) == 0) {
            count++;
        }
        return count;
    }

public:
    HyperLogLog(int b_bits = 10, HashFuncGen::HashType hash_type = HashFuncGen::MURMUR) : b(b_bits), m(1 << b_bits), registers(m, 0), hash_func(hash_type) {
        alpha_m = get_alpha_m();
    }

    void add(const std::string& element) {
        uint32_t hash = hash_func(element);
        uint32_t index = extract_index(hash, b);
        uint8_t zeros = count_leading_zeros(hash, b);

        if (zeros > registers[index]) {
            registers[index] = zeros;
        }
    }

    void add_stream(const std::vector<std::string>& stream) {
        for (const auto& element : stream) {
            add(element);
        }
    }

    double estimate() const {
        double sum = 0.0;
        int zero_registers = 0;

        for (int i = 0; i < m; ++i) {
            sum += 1.0 / (1ULL << registers[i]);
            if (registers[i] == 0) {
                zero_registers++;
            }
        }

        double raw_estimate = alpha_m * m * m / sum;

        if (raw_estimate <= 2.5 * m) {
            if (zero_registers > 0) {
                raw_estimate = m * std::log(static_cast<double>(m) / zero_registers);
            }
        } else if (raw_estimate > (1.0 / 30.0) * (1ULL << 32)) {
            raw_estimate = - (1ULL << 32) * std::log(1.0 - raw_estimate / (1ULL << 32));
        }

        return raw_estimate;
    }

    void reset() {
        std::fill(registers.begin(), registers.end(), 0);
    }
};

struct StreamAnalysisResult {
    std::string stream_name;
    std::vector<int> time_points;
    std::vector<int> exact_counts;
    std::vector<double> estimates;
    std::vector<double> mean_estimates;
    std::vector<double> std_deviations;
    double mean_error_percent;
    double max_error_percent;
    double avg_std_deviation;
};

StreamAnalysisResult analyze_stream(
    const std::string& stream_name,
    const std::vector<std::string>& stream,
    int b_bits = 10) {

    StreamAnalysisResult result;
    result.stream_name = stream_name;

    RandomStreamGen temp_gen(0.25, 100000, 12345);

    auto time_slices = temp_gen.split_equal_parts(stream, 10);
    HyperLogLog hll(b_bits);

    std::unordered_set<std::string> cumulative_exact;
    std::vector<double> running_estimates;

    for (size_t i = 0; i < time_slices.size(); ++i) {
        hll.add_stream(time_slices[i]);

        for (const auto& str : time_slices[i]) {
            cumulative_exact.insert(str);
        }

        int exact = cumulative_exact.size();
        double estimate = hll.estimate();

        result.time_points.push_back((i + 1) * 10);
        result.exact_counts.push_back(exact);
        result.estimates.push_back(estimate);

        running_estimates.push_back(estimate);

        double mean_estimate = 0.0;
        double std_deviation = 0.0;

        if (!running_estimates.empty()) {
            mean_estimate = std::accumulate(running_estimates.begin(),running_estimates.end(), 0.0) / running_estimates.size();

            double sq_sum = 0.0;
            for (double est : running_estimates) {
                double diff = est - mean_estimate;
                sq_sum += diff * diff;
            }
            std_deviation = std::sqrt(sq_sum / running_estimates.size());
        }

        result.mean_estimates.push_back(mean_estimate);
        result.std_deviations.push_back(std_deviation);
    }

    double total_error = 0.0;
    double max_error = 0.0;
    double total_std = 0.0;

    for (size_t i = 0; i < result.exact_counts.size(); ++i) {
        if (result.exact_counts[i] > 0) {
            double error = std::abs(result.estimates[i] - result.exact_counts[i]) / result.exact_counts[i] * 100.0;
            total_error += error;
            if (error > max_error) max_error = error;
        }
        total_std += result.std_deviations[i];
    }

    result.mean_error_percent = total_error / result.exact_counts.size();
    result.max_error_percent = max_error;
    result.avg_std_deviation = total_std / result.std_deviations.size();

    return result;
}

void save_stream_data(const StreamAnalysisResult& result) {
    std::string filename = result.stream_name + "_plot_data.csv";
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Ошибка открытия файла: " << filename << std::endl;
        return;
    }

    file << "time_point,exact_count,estimate,mean_estimate,std_deviation\n";

    for (size_t i = 0; i < result.time_points.size(); ++i) {
        file << result.time_points[i] << ","
             << result.exact_counts[i] << ","
             << std::fixed << std::setprecision(2) << result.estimates[i] << ","
             << result.mean_estimates[i] << ","
             << result.std_deviations[i] << "\n";
    }

    file.close();
    std::cout << "  Данные сохранены в " << filename << std::endl;
}

void save_aggregated_stats(const std::vector<StreamAnalysisResult>& results) {
    std::ofstream file("aggregated_stats.csv");

    if (!file.is_open()) {
        std::cerr << "Ошибка открытия файла: aggregated_stats.csv" << std::endl;
        return;
    }

    file << "stream_name,mean_error_percent,max_error_percent,std_deviation\n";

    for (const auto& result : results) {
        file << result.stream_name << ","
             << std::fixed << std::setprecision(2) << result.mean_error_percent << ","
             << result.max_error_percent << ","
             << result.avg_std_deviation << "\n";
    }

    file.close();
    std::cout << "\nАгрегированные статистики сохранены в aggregated_stats.csv" << std::endl;
}

void generate_test_data() {
    std::cout << "Генерация тестовых данных для HyperLogLog анализа.\n";

    RandomStreamGen gen_small(0.3, 10000, 12345);     // Маленький поток: 30% дубликатов
    RandomStreamGen gen_medium(0.15, 50000, 23456);   // Средний поток: 15% дубликатов
    RandomStreamGen gen_large(0.05, 200000, 34567);   // Большой поток: 5% дубликатов

    std::vector<StreamAnalysisResult> all_results;

    {
        std::cout << "\n1. Анализ маленького потока (5000 элементов, 30% дубликатов).\n";
        auto stream = gen_small.generate_stream(5000);
        auto result = analyze_stream("stream_small", stream, 10);

        std::cout << "   Средняя ошибка: " << std::fixed << std::setprecision(2)
                  << result.mean_error_percent << "%\n";
        std::cout << "   Максимальная ошибка: " << result.max_error_percent << "%\n";
        std::cout << "   Среднее std deviation: " << result.avg_std_deviation << "\n";

        save_stream_data(result);
        all_results.push_back(result);
    }

    {
        std::cout << "\n2. Анализ среднего потока (20000 элементов, 15% дубликатов).\n";
        auto stream = gen_medium.generate_stream(20000);
        auto result = analyze_stream("stream_medium", stream, 10);

        std::cout << "   Средняя ошибка: " << std::fixed << std::setprecision(2)
                  << result.mean_error_percent << "%\n";
        std::cout << "   Максимальная ошибка: " << result.max_error_percent << "%\n";
        std::cout << "   Среднее std deviation: " << result.avg_std_deviation << "\n";

        save_stream_data(result);
        all_results.push_back(result);
    }

    {
        std::cout << "\n3. Анализ большого потока (100000 элементов, 5% дубликатов).\n";
        auto stream = gen_large.generate_stream(100000);
        auto result = analyze_stream("stream_large", stream, 10);

        std::cout << "   Средняя ошибка: " << std::fixed << std::setprecision(2)
                  << result.mean_error_percent << "%\n";
        std::cout << "   Максимальная ошибка: " << result.max_error_percent << "%\n";
        std::cout << "   Среднее std deviation: " << result.avg_std_deviation << "\n";

        save_stream_data(result);
        all_results.push_back(result);
    }

    save_aggregated_stats(all_results);

    std::cout << "Созданные файлы:\n";
    std::cout << "stream_small_plot_data.csv\n";
    std::cout << "stream_medium_plot_data.csv\n";
    std::cout << "stream_large_plot_data.csv\n";
    std::cout << "aggregated_stats.csv\n";
}

int main() {
    generate_test_data();

    return 0;
}
