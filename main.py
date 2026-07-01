"""Provider Inspector — entry point."""

from ui.app import ProviderInspectorApp


def main() -> None:
    app = ProviderInspectorApp()
    app.run()


if __name__ == "__main__":
    main()
