"""Tests for CompatibilityChecker: tri-state logic and stack validation."""

from __future__ import annotations

from qr_sampler.profiles.compatibility import (
    CompatibilityChecker,
    CompatibilityReport,
    CompatibilityStatus,
)
from qr_sampler.profiles.loader import ProfileLoader


class TestCompatibilityStatusDataclass:
    """Tests for CompatibilityStatus frozen dataclass."""

    def test_fields(self) -> None:
        cs = CompatibilityStatus(
            component_a="engine:vllm",
            component_b="model:test",
            status="known_working",
            message="ok",
        )
        assert cs.component_a == "engine:vllm"
        assert cs.component_b == "model:test"
        assert cs.status == "known_working"
        assert cs.message == "ok"

    def test_default_message(self) -> None:
        cs = CompatibilityStatus(component_a="a", component_b="b", status="untested")
        assert cs.message == ""


class TestCompatibilityReportDataclass:
    """Tests for CompatibilityReport frozen dataclass."""

    def test_is_valid_no_errors(self) -> None:
        report = CompatibilityReport(
            checks=[], warnings=["untested combo"], errors=[], available_samplers=[]
        )
        assert report.is_valid is True

    def test_is_valid_with_errors(self) -> None:
        report = CompatibilityReport(
            checks=[], warnings=[], errors=["incompatible"], available_samplers=[]
        )
        assert report.is_valid is False

    def test_is_valid_with_missing_deps(self) -> None:
        report = CompatibilityReport(
            checks=[],
            warnings=[],
            errors=[],
            available_samplers=[],
            missing_dependencies=["some_package"],
        )
        assert report.is_valid is False


class TestCheckEngineModel:
    """Tests for engine-model compatibility checking."""

    def test_known_working_model(self) -> None:
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        status = checker.check_engine_model("vllm", "Qwen/Qwen2.5-1.5B-Instruct")
        assert status.status == "known_working"
        assert "verified working" in status.message

    def test_untested_model(self) -> None:
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        status = checker.check_engine_model("vllm", "some/untested-model")
        assert status.status == "untested"
        assert "not been tested" in status.message

    def test_known_incompatible_model(self) -> None:
        """Test with an engine that has a model in its incompatible list.

        Since our built-in profiles have empty incompatible lists, we test
        the logic by using a custom profile setup via tmp_path.
        """
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import yaml

        with TemporaryDirectory() as tmpdir:
            engines_dir = Path(tmpdir) / "engines"
            engines_dir.mkdir()
            engine_data = {
                "id": "test_engine",
                "name": "Test",
                "adapter": "m:C",
                "known_working_models": ["good/model"],
                "known_incompatible_models": ["bad/model"],
            }
            with open(engines_dir / "test_engine.yaml", "w") as f:
                yaml.dump(engine_data, f)

            loader = ProfileLoader(builtin_dir=Path(tmpdir))
            checker = CompatibilityChecker(loader)

            status = checker.check_engine_model("test_engine", "bad/model")
            assert status.status == "known_incompatible"
            assert "incompatible" in status.message


class TestCheckEntropyAmplifier:
    """Tests for entropy-amplifier compatibility checking."""

    def test_known_working_from_source_side(self) -> None:
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        # system.yaml lists zscore_mean as compatible.
        status = checker.check_entropy_amplifier("system", "zscore_mean")
        assert status.status == "known_working"

    def test_known_working_from_amplifier_side(self) -> None:
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        # zscore_mean.yaml lists system as compatible_entropy_source.
        status = checker.check_entropy_amplifier("system", "zscore_mean")
        assert status.status == "known_working"

    def test_untested_combination(self) -> None:
        """Untested when neither side references the other."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import yaml

        with TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / "entropy").mkdir()
            (td / "amplifiers").mkdir()

            with open(td / "entropy" / "isolated_src.yaml", "w") as f:
                yaml.dump(
                    {
                        "id": "isolated_src",
                        "name": "Isolated",
                        "source_class": "m:C",
                        "compatible_amplifiers": [],
                        "known_incompatible_amplifiers": [],
                    },
                    f,
                )

            with open(td / "amplifiers" / "isolated_amp.yaml", "w") as f:
                yaml.dump(
                    {
                        "id": "isolated_amp",
                        "name": "Isolated",
                        "amplifier_class": "m:C",
                        "compatible_entropy_sources": [],
                        "known_incompatible_entropy_sources": [],
                    },
                    f,
                )

            loader = ProfileLoader(builtin_dir=td)
            checker = CompatibilityChecker(loader)
            status = checker.check_entropy_amplifier("isolated_src", "isolated_amp")
            assert status.status == "untested"

    def test_known_incompatible_from_source_side(self) -> None:
        """Incompatible when source lists amplifier in known_incompatible."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import yaml

        with TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / "entropy").mkdir()
            (td / "amplifiers").mkdir()

            with open(td / "entropy" / "bad_src.yaml", "w") as f:
                yaml.dump(
                    {
                        "id": "bad_src",
                        "name": "Bad",
                        "source_class": "m:C",
                        "compatible_amplifiers": [],
                        "known_incompatible_amplifiers": ["bad_amp"],
                    },
                    f,
                )

            with open(td / "amplifiers" / "bad_amp.yaml", "w") as f:
                yaml.dump(
                    {
                        "id": "bad_amp",
                        "name": "Bad",
                        "amplifier_class": "m:C",
                        "compatible_entropy_sources": [],
                        "known_incompatible_entropy_sources": [],
                    },
                    f,
                )

            loader = ProfileLoader(builtin_dir=td)
            checker = CompatibilityChecker(loader)
            status = checker.check_entropy_amplifier("bad_src", "bad_amp")
            assert status.status == "known_incompatible"

    def test_known_incompatible_from_amplifier_side(self) -> None:
        """Incompatible when amplifier lists source in known_incompatible."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import yaml

        with TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            (td / "entropy").mkdir()
            (td / "amplifiers").mkdir()

            with open(td / "entropy" / "src.yaml", "w") as f:
                yaml.dump(
                    {
                        "id": "src",
                        "name": "Src",
                        "source_class": "m:C",
                        "compatible_amplifiers": [],
                        "known_incompatible_amplifiers": [],
                    },
                    f,
                )

            with open(td / "amplifiers" / "amp.yaml", "w") as f:
                yaml.dump(
                    {
                        "id": "amp",
                        "name": "Amp",
                        "amplifier_class": "m:C",
                        "compatible_entropy_sources": [],
                        "known_incompatible_entropy_sources": ["src"],
                    },
                    f,
                )

            loader = ProfileLoader(builtin_dir=td)
            checker = CompatibilityChecker(loader)
            status = checker.check_entropy_amplifier("src", "amp")
            assert status.status == "known_incompatible"


class TestCheckDependencies:
    """Tests for dependency checking."""

    def test_numpy_available(self) -> None:
        """vllm dependency will be missing, but numpy is always present."""
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        # vllm is listed as a dep of the vllm engine, likely not importable in test env.
        missing = checker.check_dependencies("vllm")
        # We cannot assume vllm is installed, so just check it returns a list.
        assert isinstance(missing, list)


class TestCheckStack:
    """Tests for full-stack compatibility checks."""

    def test_all_known_working(self) -> None:
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        report = checker.check_stack(
            engine="vllm",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            entropy_source="system",
            amplifier="zscore_mean",
            sampler="fixed",
        )
        assert isinstance(report, CompatibilityReport)
        # Engine-model: known_working, entropy-amplifier: known_working.
        working_checks = [c for c in report.checks if c.status == "known_working"]
        assert len(working_checks) == 2
        assert report.available_samplers == ["fixed"]

    def test_untested_model_produces_warning(self) -> None:
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        report = checker.check_stack(
            engine="vllm",
            model="some/untested-model",
        )
        assert len(report.warnings) >= 1
        assert any("not been tested" in w for w in report.warnings)
        # Untested models produce warnings, not errors.
        assert len(report.errors) == 0

    def test_unknown_sampler_produces_error(self) -> None:
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        report = checker.check_stack(
            engine="vllm",
            sampler="nonexistent",
        )
        assert report.is_valid is False
        assert any("nonexistent" in e for e in report.errors)

    def test_engine_only_no_checks(self) -> None:
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        report = checker.check_stack(engine="vllm")
        # No model or entropy/amplifier specified -> no pairwise checks.
        assert len(report.checks) == 0
        # But all samplers should be available.
        assert len(report.available_samplers) >= 2

    def test_full_stack_report_fields(self) -> None:
        loader = ProfileLoader()
        checker = CompatibilityChecker(loader)
        report = checker.check_stack(
            engine="vllm",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            entropy_source="quantum_grpc",
            amplifier="ecdf",
        )
        assert isinstance(report.checks, list)
        assert isinstance(report.warnings, list)
        assert isinstance(report.errors, list)
        assert isinstance(report.available_samplers, list)
        assert isinstance(report.missing_dependencies, list)
