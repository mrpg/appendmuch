import pytest
from driver_contract import DriverContract

from appendmuch import Memory


class TestMemoryDriver(DriverContract):
    @pytest.fixture
    def driver_factory(self):
        instances = {}

        def factory(name="default"):
            return instances.setdefault(name, Memory())

        return factory

    def test_size_is_none(self, driver_factory):
        assert driver_factory().size() is None
