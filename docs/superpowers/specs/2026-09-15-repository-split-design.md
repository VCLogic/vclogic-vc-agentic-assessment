# Repository split

Approved by the user on 2026-09-15.

The pipeline repository is https://github.com/VCLogic/vclogic-vc-agentic-assessment. It owns the existing vc_clone_graph Python package, excluding its web subpackage; assessment and rehearsal CLIs; configurations; investor inputs; evaluation utilities; and engine tests.

The application repository is https://github.com/VCLogic/vclogic-web-application. It owns the React frontend, a separate vclogic_web Python package, API/session/project orchestration, and web tests. It depends on the pipeline distribution. The API receives an explicit pipeline workspace for data, configuration, and runtime artifacts. Frontend assets remain relative to the application checkout.

Copy the current working files, including uncommitted source improvements. Do not modify the source repository. Exclude secrets, environments, generated indexes, checkpoints, runtime outputs, build products, and caches. Preserve reusable evaluation reference inputs. Existing historical output-dependent checks may skip when outputs are absent.

Validate package builds, dependency direction, deterministic pipeline execution, engine tests, API tests, and frontend unit tests/build. Do not invoke paid model calls. Local commits prepare both repositories for publication; pushing is a separate action.
