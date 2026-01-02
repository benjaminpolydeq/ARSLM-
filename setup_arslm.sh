#!/data/data/com.termux/files/usr/bin/bash
# Setup SSH et dépôt ARSLM sur Termux
set -e

EMAIL="kbenjio9@gmail.com"
GITHUB_USER="benjaminpolydeq"
REPO_NAME="ARSLM"

echo "📌 Installation d'OpenSSH..."
pkg install openssh -y

# Génération clé SSH
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
    echo "🔑 Génération clé SSH..."
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    ssh-keygen -t ed25519 -C "$EMAIL" -f ~/.ssh/id_ed25519 -N ""
else
    echo "⚠ Clé SSH existante détectée."
fi

# Configuration SSH
cat > ~/.ssh/config << EOF
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
EOF
chmod 600 ~/.ssh/config

# Affichage clé publique
echo ""
echo "📋 COPIEZ CETTE CLÉ PUBLIQUE ET AJOUTEZ-LA SUR GITHUB:"
echo "https://github.com/settings/keys"
echo ""
cat ~/.ssh/id_ed25519.pub
echo ""

read -p "Appuyez sur Entrée après avoir ajouté la clé sur GitHub..."

# Configuration dépôt
cd ~/ARSLM || git clone git@github.com:${GITHUB_USER}/${REPO_NAME}.git && cd ${REPO_NAME}
git remote set-url origin git@github.com:${GITHUB_USER}/${REPO_NAME}.git

# Test push
if git status --porcelain | grep -q .; then
    git add .
    git commit -m "chore: Setup SSH GitHub Termux"
fi

echo "🚀 Test de push..."
git push origin main || echo "⚠ Vérifiez votre clé SSH et la configuration du dépôt"

echo "✅ Configuration terminée !"