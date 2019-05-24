#!/usr/bin/env bash

INSTALL_DIR="/usr/local"

rm -drf $INSTALL_DIR/include/IMAGE
rm -drf $INSTALL_DIR/lib/IMAGE
rm -drf $INSTALL_DIR/bin/IMAGE
rm -drf $INSTALL_DIR/share/IMAGE
#需要加条件 移除成功才显示
echo "移除文件夹:$INSTALL_DIR/include/SFM"
echo "移除文件夹:$INSTALL_DIR/lib/SFM"
echo "移除文件夹:$INSTALL_DIR/bin/SFM"
echo "移除文件夹:$INSTALL_DIR/share/SFM"