#!/bin/bash
# Publication script for ARSLM package to PyPI

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if in virtual environment
check_venv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "Not in a virtual environment. It's recommended to use one."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Clean old builds
clean_builds() {
    print_header "Cleaning old builds"
    
    rm -rf build/ dist/ *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    print_success "Cleaned old builds"
}

# Install dependencies
install_deps() {
    print_header "Installing dependencies"
    
    pip install --upgrade pip setuptools wheel
    pip install --upgrade build twine
    pip install -e ".[dev]"
    
    print_success "Dependencies installed"
}

# Run tests
run_tests() {
    print_header "Running tests"
    
    python -m pytest tests/ -v --cov=arslm --cov-report=term-missing
    
    if [ $? -ne 0 ]; then
        print_error "Tests failed!"
        exit 1
    fi
    
    print_success "All tests passed"
}

# Code quality checks
check_code_quality() {
    print_header "Checking code quality"
    
    print_info "Running black..."
    python -m black --check arslm/ tests/
    
    if [ $? -ne 0 ]; then
        print_warning "Code formatting issues found. Run: black arslm/ tests/"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_info "Running isort..."
    python -m isort --check arslm/ tests/
    
    if [ $? -ne 0 ]; then
        print_warning "Import sorting issues found. Run: isort arslm/ tests/"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_info "Running flake8..."
    python -m flake8 arslm/ tests/ --max-line-length=88 --extend-ignore=E203,W503
    
    if [ $? -ne 0 ]; then
        print_warning "Linting issues found"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_success "Code quality checks passed"
}

# Build package
build_package() {
    print_header "Building package"
    
    python -m build
    
    if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
        print_error "Build failed - no dist directory or empty"
        exit 1
    fi
    
    print_success "Package built successfully"
    
    # Show built files
    print_info "Built files:"
    ls -lh dist/
}

# Check package
check_package() {
    print_header "Checking package"
    
    python -m twine check dist/*
    
    if [ $? -ne 0 ]; then
        print_error "Package check failed!"
        exit 1
    fi
    
    print_success "Package check passed"
}

# Get version from package
get_version() {
    python -c "import arslm; print(arslm.__version__)"
}

# Check if version exists on PyPI
check_version_exists() {
    local version=$1
    local url="https://pypi.org/pypi/arslm/$version/json"
    
    if curl --output /dev/null --silent --head --fail "$url"; then
        return 0  # Version exists
    else
        return 1  # Version doesn't exist
    fi
}

# Upload to TestPyPI
upload_testpypi() {
    print_header "Uploading to TestPyPI"
    
    print_warning "This will upload to TestPyPI (test.pypi.org)"
    print_info "Make sure you have configured your TestPyPI token in ~/.pypirc"
    
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi
    
    python -m twine upload --repository testpypi dist/*
    
    if [ $? -eq 0 ]; then
        print_success "Uploaded to TestPyPI successfully!"
        
        local version=$(get_version)
        print_info "Test installation with:"
        echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arslm==$version"
    else
        print_error "Upload to TestPyPI failed!"
        exit 1
    fi
}

# Upload to PyPI
upload_pypi() {
    print_header "Uploading to PyPI (Production)"
    
    local version=$(get_version)
    
    print_warning "⚠️  PRODUCTION UPLOAD ⚠️"
    print_warning "This will upload version $version to PyPI (pypi.org)"
    print_warning "This action CANNOT be undone!"
    
    # Check if version already exists
    if check_version_exists "$version"; then
        print_error "Version $version already exists on PyPI!"
        print_info "You need to bump the version in arslm/__init__.py"
        exit 1
    fi
    
    print_info "Version $version does not exist on PyPI (good!)"
    
    echo
    read -p "Are you ABSOLUTELY sure? Type 'yes' to continue: " -r
    echo
    if [[ ! $REPLY == "yes" ]]; then
        print_info "Upload cancelled"
        return
    fi
    
    python -m twine upload dist/*
    
    if [ $? -eq 0 ]; then
        print_success "🎉 Successfully uploaded to PyPI!"
        print_success "Package is now available at: https://pypi.org/project/arslm/"
        print_info "Installation command:"
        echo "  pip install arslm==$version"
        
        # Create git tag
        print_info "Creating git tag..."
        git tag -a "v$version" -m "Release version $version"
        
        print_info "Push tag with:"
        echo "  git push origin v$version"
    else
        print_error "Upload to PyPI failed!"
        exit 1
    fi
}

# Show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --test       Upload to TestPyPI only"
    echo "  --prod       Upload to PyPI (production)"
    echo "  --no-test    Skip running tests"
    echo "  --no-check   Skip code quality checks"
    echo "  --help       Show this help message"
    echo ""
    echo "Default: Run all checks and upload to TestPyPI"
}

# Main execution
main() {
    local upload_target="test"
    local run_tests=true
    local run_checks=true
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --test)
                upload_target="test"
                shift
                ;;
            --prod)
                upload_target="prod"
                shift
                ;;
            --no-test)
                run_tests=false
                shift
                ;;
            --no-check)
                run_checks=false
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Start
    print_header "ARSLM Publication Script"
    
    # Show current version
    local version=$(get_version)
    print_info "Current version: $version"
    
    # Check virtual environment
    check_venv
    
    # Clean builds
    clean_builds
    
    # Install dependencies
    install_deps
    
    # Run tests
    if [ "$run_tests" = true ]; then
        run_tests
    else
        print_warning "Skipping tests"
    fi
    
    # Code quality
    if [ "$run_checks" = true ]; then
        check_code_quality
    else
        print_warning "Skipping code quality checks"
    fi
    
    # Build
    build_package
    
    # Check package
    check_package
    
    # Upload
    if [ "$upload_target" = "test" ]; then
        upload_testpypi
    elif [ "$upload_target" = "prod" ]; then
        upload_pypi
    fi
    
    print_header "Process completed!"
}

# Run main
main "$@"