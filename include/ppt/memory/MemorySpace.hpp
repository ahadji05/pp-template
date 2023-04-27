/**
 * @file
 *
 * @author  Andreas Hadjigeorgiou, The Cyprus Institute,
 *          Personal-site: https://ahadji05.github.io,
 *          E-mail: a.hadjigeorgiou@cyi.ac.cy
 *
 * @copyright 2022 CaSToRC (The Cyprus Institute), Delphi Consortium (TU Delft)
 *
 * @version 1.0
 *
 * @section LICENCE
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef PPT_MEMORY_SPACE_HPP
#define PPT_MEMORY_SPACE_HPP

#include <iostream>
#include <string>

namespace ppt
{

/**
 * @brief Base MemorySpase class from which all the Memory-Spaces are derived.
 * The member variables are used for controling whether memory allocations,
 * copies, or releases receive a run-time error.
 */
class MemorySpaceBase
{
  public:
    /**
     * @brief Four different messages can be returned from a MemorySpace
     * related operation.
     */
    enum class message
    {
        no_error,
        allocation_failed,
        release_failed,
        copying_failed
    };

    /**
     * @brief Routine for printing in std::ostream a message that is associated
     * with a tag, for each of the four possible return messages.
     *
     * @tparam T std::ostream-compatible type for the tag.
     * @param message_type A valid message type defined in the enum class
     * message.
     * @param tag Tag associated with the message that serves identification
     * purposes.
     */
    template <typename T> static void print_message(message message_type, T tag)
    {
        switch (message_type)
        {
        case message::no_error: return;

        case message::allocation_failed: std::cout << "Memory allocation with tag (" << tag << ") failed!\n"; return;

        case message::copying_failed: std::cout << "Memory copying with tag (" << tag << ") failed!\n"; return;

        case message::release_failed: std::cout << "Memory release with tag (" << tag << ") failed!\n"; return;

        default: std::cout << "Error message_type with tag (" << tag << ")is undefined!\n"; return;
        }
    }

    /**
     * @brief Routine for printing in std::ostream a message that is associated
     * with tag = "default".
     *
     * @param message_type A valid message type defined in the enum class
     * message.
     */
    static void print_message(message message_type)
    {
        print_message(message_type, "default");
    }

    /**
     * @brief Aliasing of message type, used as return parameter in the
     * memory spaces derived from this Base class.
     */
    using return_t = message;
};

} // namespace ppt

#endif // PPT_MEMORY_SPACE_HPP
