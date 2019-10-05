import configparser
import os
from copy import deepcopy


def family(current_class):
    yield current_class
    for next_class in current_class.__bases__:
        for val in family(next_class):
            yield val


class NamedConfig(object):
    """Parses generic config"""

    def __init__(self, config_path):
        """
        Parameters
        ----------
        config_path: String
            The path to the config file
        """
        self.path = config_path
        self.parser = configparser.ConfigParser()
        self.parser.read(self.path)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, v if k == "parser" else deepcopy(v, memo))
        return result

    def __eq__(self, other):
        return isinstance(other, NamedConfig) and self.path == other.path

    def get(self, section_name, attribute_name, attribute_type=str):
        """

        Parameters
        ----------
        section_name
        attribute_type: type
            The type to which the value should be cast
        attribute_name: String
            The analysis_path of the attribute

        Returns
        -------
        prior_array: []
            An array describing a prior
        """
        try:
            string_value = self.parser.get(section_name, attribute_name)
        except configparser.NoSectionError:
            raise configparser.NoSectionError(
                "Could not find section {} in config at path {}".format(
                    section_name, self.path
                )
            )
        except configparser.NoOptionError as e:
            raise configparser.NoOptionError(
                "could not find option {} in section {} of config at path {}".format(
                    attribute_name, section_name, self.path
                ),
                e.section,
            )
        if string_value == "None":
            return None
        if attribute_type is bool:
            return string_value == "True"
        return attribute_type(string_value)

    def has(self, section_name, attribute_name):
        """
        Parameters
        ----------
        section_name
        attribute_name: String
            The analysis_path of the attribute

        Returns
        -------
        has_prior: bool
            True iff a prior exists for the module, class and attribute
        """
        return self.parser.has_option(section_name, attribute_name)


class Config(object):
    def __init__(self, config_path, output_path="output"):
        self.config_path = config_path
        self.non_linear = NamedConfig("{}/non_linear.ini".format(config_path))
        self.label_format = NamedConfig("{}/label_format.ini".format(config_path))
        self.general = NamedConfig("{}/general.ini".format(config_path))
        self.visualize = NamedConfig("{}/visualize.ini".format(config_path))
        self.output_path = output_path


def is_config_in(folder):
    return os.path.isdir("{}/config".format(folder))


"""
Search for default configuration and put output in the same folder as config.

The search is performed in this order:
1) autolens_workspace. This is assumed to be in the same directory as autolens in the Docker 
   container
2) current working directory. This is to allow for installation and use with pip where 
   users would expect the configuration in their current directory to be used.
3) relative. This is a backup for when no configuration is found. In this case it is 
   still assumed a autolens_workspace directory exists in the same directory as autofit.
"""

autofit_directory = os.path.dirname(os.path.realpath(__file__))
docker_workspace_directory = "/home/user/autolens_workspace"
current_directory = os.getcwd()

try:
    workspace_path = os.environ["WORKSPACE"]
    default = Config(
        "{}/config".format(workspace_path), "{}/output/".format(workspace_path)
    )
except KeyError:
    if is_config_in(docker_workspace_directory):
        CONFIG_PATH = "{}/config".format(docker_workspace_directory)
        default = Config(CONFIG_PATH, "{}/output/".format(docker_workspace_directory))
    elif is_config_in(current_directory):
        CONFIG_PATH = "{}/config".format(current_directory)
        default = Config(CONFIG_PATH, "{}/output/".format(current_directory))
    elif is_config_in("{}/../..".format(current_directory)):
        CONFIG_PATH = "{}/../../config".format(current_directory)
        default = Config(CONFIG_PATH, "{}/output/".format(current_directory))
    elif is_config_in("{}/../autolens_workspace".format(current_directory)):
        CONFIG_PATH = "{}/../autolens_workspace/config".format(current_directory)
        default = Config(
            CONFIG_PATH, "{}/../autolens_workspace/output/".format(current_directory)
        )
    else:
        CONFIG_PATH = "{}/../autolens_workspace/config".format(autofit_directory)
        default = Config(
            CONFIG_PATH, "{}/../autolens_workspace/output/".format(autofit_directory)
        )

instance = default
